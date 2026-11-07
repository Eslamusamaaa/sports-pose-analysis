"""Real-time webcam pipeline: OpenPose (live camera) -> skeleton -> ST-GCN.

DESIGN NOTE - why OpenPose owns the camera
------------------------------------------
Streaming browser frames into OpenPose would require its Python API, but the
portable release only ships ``pyopenpose.cp37-win_amd64.pyd`` - a CPython 3.7
ABI extension that cannot be imported by the 3.11/3.12 interpreters here, and
3.7 is end-of-life. Instead OpenPoseDemo.exe is launched ONCE with ``--camera``
and left running; it writes one JSON (and one rendered JPEG) per frame, which a
watcher thread consumes. Consequences:

* The camera is opened by the OpenPose process on the machine running the
  backend. Since backend and browser are the same host here, that is still the
  user's webcam - and it avoids an encode/transport hop, so latency is lower.
* A webcam can normally only be opened by one process at a time, so the browser
  must NOT also call getUserMedia on the same device; the live view is fed by
  OpenPose's rendered frames instead.

Threading model: a watcher thread owns all OpenPose I/O and mutates a lock-
guarded snapshot; the WebSocket handler just reads that snapshot on a timer.
This keeps the async layer free of blocking file and subprocess calls.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger
from .keypoints import COCO_FROM_BODY25
from .skeleton_service import NUM_BODY25, VALUES_PER_JOINT, limb_step

WORKER = BACKEND_DIR / "mmaction_worker.py"
MSG_PREFIX = "@@MMA@@"


class RealtimeError(RuntimeError):
    pass


class LiveTracker:
    """Incremental nearest-centroid tracker.

    The batch tracker in skeleton_service works over a complete video; streaming
    needs frame-at-a-time assignment. Same limitations apply: no re-identification
    and no occlusion handling.
    """

    def __init__(self, gate: float = 160.0, max_missing: int = 15) -> None:
        self.gate = gate
        self.max_missing = max_missing
        self._next_id = 0
        self._tracks: dict[int, dict] = {}  # tid -> {centroid, missing, hits}

    def update(self, people: list[dict]) -> None:
        """Assign a 'track_id' to each person dict in place."""
        pairs: list[tuple[float, int, int]] = []
        for i, p in enumerate(people):
            if p["centroid"] is None:
                continue
            for tid, tr in self._tracks.items():
                pairs.append((math.dist(p["centroid"], tr["centroid"]), i, tid))
        pairs.sort()

        taken_p: set[int] = set()
        taken_t: set[int] = set()
        assign: dict[int, int] = {}
        for dist, i, tid in pairs:
            if i in taken_p or tid in taken_t or dist > self.gate:
                continue
            assign[i] = tid
            taken_p.add(i)
            taken_t.add(tid)

        for i, p in enumerate(people):
            if p["centroid"] is None:
                p["track_id"] = None
                continue
            tid = assign.get(i)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"centroid": p["centroid"], "missing": 0, "hits": 0}
            p["track_id"] = tid
            tr = self._tracks[tid]
            tr["centroid"] = p["centroid"]
            tr["missing"] = 0
            tr["hits"] += 1

        for tid in list(self._tracks):
            if tid not in taken_t and tid not in assign.values():
                self._tracks[tid]["missing"] += 1
                if self._tracks[tid]["missing"] > self.max_missing:
                    del self._tracks[tid]

    def best_tracks(self, n: int) -> list[int]:
        ranked = sorted(self._tracks.items(), key=lambda kv: kv[1]["hits"], reverse=True)
        return [tid for tid, _ in ranked[:n]]


def _decode_reid(b64: str | None):
    """Live re-id vector (float16, base64 in the per-frame JSON) -> np array."""
    if not b64:
        return None
    import base64

    import numpy as np

    v = np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _parse_people(raw_people: list[dict]) -> list[dict]:
    out = []
    for raw in raw_people:
        flat = raw.get("pose_keypoints_2d") or []
        if len(flat) != NUM_BODY25 * VALUES_PER_JOINT:
            continue
        joints = [
            (float(flat[i * 3]), float(flat[i * 3 + 1]), float(flat[i * 3 + 2]))
            for i in range(NUM_BODY25)
        ]
        pts = [(x, y) for x, y, c in joints if c > 0.1 and (x or y)]
        centroid = (
            (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            if pts
            else None
        )
        confs = [c for _, _, c in joints if c > 0]
        pid = raw.get("person_id")
        out.append(
            {
                "keypoints": joints,
                "centroid": centroid,
                "mean_confidence": (sum(confs) / len(confs)) if confs else 0.0,
                "valid_keypoints": sum(1 for _, _, c in joints if c > 0.1),
                # YOLO11 writes its ByteTrack id here (OpenPose: -1).
                "engine_id": pid[0] if isinstance(pid, list) and pid and pid[0] >= 0 else None,
                "bbox": raw.get("bbox"),
                "appearance": raw.get("appearance"),
                "reid": _decode_reid(raw.get("reid_b64")),
            }
        )
    return out


class PlayerSelector:
    """Pick the live "players": the people who are present AND moving.

    Exponential moving averages, so a player who just walked in can take over
    within a couple of seconds, but one missed frame does not drop them. The
    same rule as batch analysis: presence x confidence x motion keeps a still
    referee or spectator from being labelled a player.
    """

    def __init__(self, alpha: float = 0.08, inherit_ratio: float = 0.2,
                 keep_s: float = 2.0, min_age_s: float = 1.0, forget_s: float = 20.0,
                 rank_by: str = "motion") -> None:
        self.alpha = alpha
        # "limbs" (karate): rank by hands/feet movement relative to the body, so
        # a coach walking about does not outrank fighters punching and kicking.
        self.rank_by = rank_by
        self.stats: dict[int, dict] = {}  # tid -> {presence, motion, last}
        # Sticky label slots: slots[0] is "Player 1" etc. A label stays on the
        # same person instead of being re-ranked every frame, and when that
        # person's id vanishes, a NEW id appearing near their last position
        # within keep_s inherits the label (the live twin of batch
        # link_player_tracklets: a fighter stepping out of frame and back).
        # Distances are a share of the frame diagonal and times are seconds
        # (live fps varies 14-32 with the engine), so nothing depends on one
        # camera or one video.
        self.inherit_ratio = inherit_ratio
        self.keep_s = keep_s
        self.min_age_s = min_age_s
        self.forget_s = forget_s
        # A 1<->2 swap needs this much sustained evidence (seconds).
        self.swap_hold_s = 0.5
        self.diag = 800.0  # replaced by the real frame diagonal once known (set_frame)
        self.slots: list[dict | None] = []  # {tid, last, missing}

    def update(self, people: list[dict]) -> None:
        seen = set()
        for p in people:
            tid = p.get("track_id")
            if tid is None or p["centroid"] is None:
                continue
            seen.add(tid)
            st = self.stats.setdefault(tid, {"presence": 0.0, "motion": 0.0, "last": None, "conf": 0.0,
                                             "first": time.time(), "limb": 0.0, "prev_kp": None})
            if st["prev_kp"] is not None and p.get("keypoints"):
                ls = limb_step(st["prev_kp"], p["keypoints"])
                if ls is not None:
                    st["limb"] += self.alpha * (ls - st["limb"])
            st["prev_kp"] = p.get("keypoints")
            step = math.dist(p["centroid"], st["last"]) if st["last"] else 0.0
            st["last"] = p["centroid"]
            st["motion"] += self.alpha * (step - st["motion"])
            st["conf"] += self.alpha * (p["mean_confidence"] - st["conf"])
        for tid, st in list(self.stats.items()):
            st["presence"] += self.alpha * ((1.0 if tid in seen else 0.0) - st["presence"])
            if st["presence"] < 0.02:
                del self.stats[tid]

    def set_frame(self, h: int, w: int) -> None:
        self.diag = math.hypot(h, w) or self.diag

    def _ranked(self) -> list[int]:
        # Only people observed long enough to have a real motion estimate: at
        # session start everyone has motion 0, and the most CONFIDENT person -
        # often a still coach - would grab the Player 1 label.
        now = time.time()
        mature = {t: st for t, st in self.stats.items() if now - st["first"] >= self.min_age_s}
        if not mature:
            return []
        key = "limb" if self.rank_by == "limbs" else "motion"
        top = max(st[key] for st in mature.values()) or 1.0
        score = {
            tid: st["presence"] * max(st["conf"], 0.05) * (0.25 + 0.75 * st[key] / top)
            for tid, st in mature.items()
        }
        return sorted(score, key=score.get, reverse=True)

    def players(self, n: int, present: dict | None = None,
                apps: dict | None = None, key: str = "appearance") -> list[int | None]:
        """Track id per label slot (index 0 = Player 1), None if absent now.

        Fixed length n, so a missing Player 1 never shifts Player 2 into the
        "Player 1" label or into slot 0 of the classifier input.
        present: {tid: centroid} for people in the current frame.
        apps: {tid: clothing signature} (RTMPose / YOLO11). When available, a
        player keeps the label by LOOKING like that player, not by keeping a
        tracker id - so an id swap in a clinch does not move the label.
        """
        present = present or {}
        while len(self.slots) < n:
            self.slots.append(None)
        if apps:
            return self._players_by_appearance(n, present, apps, key)
        taken = {s["tid"] for s in self.slots if s}
        ranked = self._ranked()
        now = time.time()
        inherit_px = self.inherit_ratio * self.diag
        for i, s in enumerate(self.slots[:n]):
            if s and s["tid"] in present:
                s["last"], s["missing"], s["seen"] = present[s["tid"]], 0, now
                continue
            if s:
                # Player's id is gone this frame: hand the label to a new id
                # close to where they were last seen, if one appeared.
                s["missing"] += 1
                near = [
                    (math.dist(c, s["last"]), tid) for tid, c in present.items()
                    if tid not in taken and math.dist(c, s["last"]) <= inherit_px
                ]
                if near:
                    _, tid = min(near)
                    taken.discard(s["tid"])
                    s.update(tid=tid, last=present[tid], missing=0, seen=now)
                    taken.add(tid)
                elif now - s.get("seen", now) > self.keep_s:
                    taken.discard(s["tid"])
                    self.slots[i] = None
                continue
            # Empty slot: best-ranked present person not already a player.
            for tid in ranked:
                if tid in present and tid not in taken:
                    self.slots[i] = {"tid": tid, "last": present[tid], "missing": 0, "seen": now}
                    taken.add(tid)
                    break
        return [s["tid"] if s and s["missing"] == 0 else None for s in self.slots[:n]]

    def _players_by_appearance(self, n: int, present: dict, apps: dict,
                               key: str = "appearance") -> list[int | None]:
        """Labels follow TRACKER IDS; the signature is only evidence.

        Same rules as batch identify_players: never hand a label to whoever
        merely looks like a player (club teammates share the team shirt), only
        (a) to a new id that appears near a player who just vanished AND looks
        like them, or (b) swap the two players' labels when the evidence that
        the tracker exchanged them persists for swap_hold_s.
        """
        import numpy as np

        from .players_identity import METRICS, _update_proto, swapped_is_better

        dist, merge_thr = METRICS[key]
        now = time.time()
        inherit_px = self.inherit_ratio * self.diag
        slots = self.slots[:n]
        taken = {s["tid"] for s in slots if s}

        for k, s in enumerate(slots):
            if s is None:
                continue
            if s["tid"] in present:
                s.update(last=present[s["tid"]], missing=0, seen=now)
                v = apps.get(s["tid"])
                if v is not None:
                    if s.get("proto") is None:
                        s["proto"] = np.array(v, np.float32)
                    elif dist(v, s["proto"]) < 0.6 * merge_thr:
                        s["proto"] = _update_proto(s["proto"], v, key, 0.05)
                continue
            s["missing"] += 1
            # (a) inherit: a new id near the last position that looks like them
            near = []
            for tid, c in present.items():
                if tid in taken or apps.get(tid) is None:
                    continue
                d_px = math.dist(c, s["last"])
                sd = dist(apps[tid], s["proto"]) if s.get("proto") is not None else 0.0
                if d_px <= inherit_px and sd <= merge_thr:
                    near.append((sd + d_px / inherit_px, tid))
            if near:
                _, tid = min(near)
                taken.discard(s["tid"])
                s.update(tid=tid, last=present[tid], missing=0, seen=now)
                taken.add(tid)
            elif now - s.get("seen", now) > self.forget_s:
                taken.discard(s["tid"])
                self.slots[k] = None

        # (b) swap check between the two players only, with a hold time
        if n >= 2 and all(self.slots[k] and self.slots[k]["missing"] == 0 for k in (0, 1)):
            s0, s1 = self.slots[0], self.slots[1]
            v0, v1 = apps.get(s0["tid"]), apps.get(s1["tid"])
            if v0 is not None and v1 is not None and s0.get("proto") is not None and s1.get("proto") is not None:
                e = swapped_is_better(v0, v1, [s0["proto"], s1["proto"]], key)
                if e > 0.1:
                    self._swap_since = getattr(self, "_swap_since", None) or now
                    if now - self._swap_since >= self.swap_hold_s:
                        s0["tid"], s1["tid"] = s1["tid"], s0["tid"]
                        s0["last"], s1["last"] = s1["last"], s0["last"]
                        self._swap_since = None
                else:
                    self._swap_since = None

        # empty slots: best limb-activity mover not already a player
        ranked = self._ranked()
        for k in range(n):
            if self.slots[k] is not None:
                continue
            for tid in ranked:
                if tid in present and tid not in taken:
                    v = apps.get(tid)
                    self.slots[k] = {"tid": tid, "last": present[tid], "missing": 0, "seen": now,
                                     "proto": np.array(v, np.float32) if v is not None else None}
                    taken.add(tid)
                    break
        return [s["tid"] if s and s["missing"] == 0 else None for s in self.slots[:n]]


def jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Return (height, width) from a JPEG's SOF marker, or None.

    Why this matters: OpenPose reports keypoints in the CAMERA's resolution, and
    PreNormalize2D rescales with (x - w/2)/(w/2). If the configured img_shape
    does not match the real camera size, every coordinate is normalised against
    the wrong extent and the predictions are quietly wrong. Reading the true
    size off the rendered frame removes that guess. Implemented by hand so the
    backend needs no imaging dependency.
    """
    i, n = 2, len(raw)
    if n < 4 or raw[0] != 0xFF or raw[1] != 0xD8:
        return None
    while i + 9 < n:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        # SOF0/1/2/3, SOF5-7, SOF9-11, SOF13-15 carry the frame dimensions.
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (raw[i + 5] << 8) | raw[i + 6]
            w = (raw[i + 7] << 8) | raw[i + 8]
            return (h, w) if h and w else None
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = (raw[i + 2] << 8) | raw[i + 3]
        if seg <= 0:
            return None
        i += 2 + seg
    return None


def _to_coco17(joints: list[tuple[float, float, float]]):
    xy, sc = [], []
    for bi in COCO_FROM_BODY25:
        x, y, c = joints[bi]
        xy.append([x, y])
        sc.append(c)
    return xy, sc


MIN_JOINTS_PRESENT = 5
# Classify only when the tracked person is in at least this share of the window.
# Without it an EMPTY room was classified: the model returned the same label
# ("nausea or vomiting condition", 0.48) for an all-zero window, and it stayed
# on screen as if it described the user.
MIN_PRESENT_SHARE = 0.5


# COCO-17 indices for the imputation below.
_NOSE, _L_SHO, _R_SHO, _L_HIP, _R_HIP = 0, 5, 6, 11, 12
_L_KNEE, _R_KNEE, _L_ANK, _R_ANK = 13, 14, 15, 16
_SEEN = 0.05


def impute_missing(frames: list[dict], slot: int = 0) -> list[dict]:
    """Fill joints OpenPose did not find, for the NTU60 live classifier.

    OpenPose reports a missed joint as (0, 0). After PreNormalize2D that is the
    TOP-LEFT CORNER of the frame, so a webcam user whose legs are out of shot
    reached the model with both feet above their head - which it read as
    "falling" while they were clapping. NTU60 (HRNet 2D) never has holes: every
    training joint has a position. So, per joint:
      1. seen in some frame of the window -> nearest-in-time real position
         (forward fill, then back fill for the leading frames);
      2. never seen (legs below the frame) -> a standing pose from body
         proportions: hips under the shoulders, knees and ankles straight down,
         each segment ~ the torso length.
    Imputed joints get a low score (0.3) so they weigh less than real ones.
    """
    import copy

    out = copy.deepcopy(frames)
    T = len(out)
    if not T:
        return out
    V = len(out[0]["keypoints"][slot])

    def seen(t: int, j: int) -> bool:
        return out[t]["keypoint_scores"][slot][j] > _SEEN

    # 1) temporal fill per joint
    for j in range(V):
        idx = [t for t in range(T) if seen(t, j)]
        if not idx or len(idx) == T:
            continue
        last = None
        for t in range(T):
            if seen(t, j):
                last = t
            elif last is not None:
                out[t]["keypoints"][slot][j] = list(out[last]["keypoints"][slot][j])
                out[t]["keypoint_scores"][slot][j] = 0.3
        first = idx[0]
        for t in range(first):
            out[t]["keypoints"][slot][j] = list(out[first]["keypoints"][slot][j])
            out[t]["keypoint_scores"][slot][j] = 0.3

    # 2) never-seen joints: standing proportions, frame by frame
    for t in range(T):
        kp, sc = out[t]["keypoints"][slot], out[t]["keypoint_scores"][slot]
        if not (sc[_L_SHO] > _SEEN and sc[_R_SHO] > _SEEN):
            continue  # no torso anchor - leave the frame as it is
        sx = (kp[_L_SHO][0] + kp[_R_SHO][0]) / 2
        sy = (kp[_L_SHO][1] + kp[_R_SHO][1]) / 2
        width = abs(kp[_L_SHO][0] - kp[_R_SHO][0]) or 1.0
        torso = 1.6 * width  # shoulder->hip is ~1.6 shoulder widths
        for hip, sho in ((_L_HIP, _L_SHO), (_R_HIP, _R_SHO)):
            if sc[hip] <= _SEEN:
                kp[hip] = [kp[sho][0] * 0.8 + sx * 0.2, sy + torso]
                sc[hip] = 0.3
        for hip, knee, ank in ((_L_HIP, _L_KNEE, _L_ANK), (_R_HIP, _R_KNEE, _R_ANK)):
            if sc[knee] <= _SEEN:
                kp[knee] = [kp[hip][0], kp[hip][1] + torso]
                sc[knee] = 0.3
            if sc[ank] <= _SEEN:
                kp[ank] = [kp[knee][0], kp[knee][1] + torso]
                sc[ank] = 0.3
    return out


def _single_or_pair(frames: list[dict]) -> list[dict]:
    """Drop the 2nd person slot unless someone is really there most of the time.

    NTU single-person clips carry no second skeleton; a mostly-empty slot (a
    brief false detection) turned solo actions into two-person classes.
    """
    if not frames:
        return frames
    # The tracker can report 1 slot in some frames and 2 in others (a second
    # track appears or expires mid-window), so pad every frame to the same
    # count before looking at slot 1 - indexing a 1-slot frame crashed live.
    m = max(len(f["keypoints"]) for f in frames)
    if m < 2:
        return frames
    frames = [
        {
            "keypoints": f["keypoints"] + [[[0.0, 0.0]] * 17] * (m - len(f["keypoints"])),
            "keypoint_scores": f["keypoint_scores"] + [[0.0] * 17] * (m - len(f["keypoints"])),
        }
        for f in frames
    ]
    second = sum(
        1 for f in frames
        if sum(1 for c in f["keypoint_scores"][1] if c > _SEEN) >= MIN_JOINTS_PRESENT
    )
    if second >= MIN_PRESENT_SHARE * len(frames):
        return frames
    return [{"keypoints": f["keypoints"][:1], "keypoint_scores": f["keypoint_scores"][:1]} for f in frames]


def person_present(entry: dict) -> bool:
    scores = entry["keypoint_scores"][0] if entry["keypoint_scores"] else []
    return sum(1 for c in scores if c > 0.05) >= MIN_JOINTS_PRESENT


class RealtimeSession:
    """One live capture session (karate page)."""

    sport = "karate"

    def __init__(self, session_dir: Path, log: JobLogger, engine: str = "openpose") -> None:
        # "openpose" | "rtmpose" | "yolo11" - who owns the camera and writes
        # the per-frame JSON/JPEG; everything downstream is engine-agnostic.
        self.engine = engine
        self.dir = session_dir
        self.log = log
        self.json_dir = session_dir / "json"
        self.img_dir = session_dir / "frames"
        for d in (self.json_dir, self.img_dir):
            d.mkdir(parents=True, exist_ok=True)

        s = get_settings()
        self.window = s.realtime_window
        self.infer_every = s.realtime_infer_every
        self.max_persons = 2

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._op: subprocess.Popen | None = None
        self._mm: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._req_id = 0

        self.buffer: deque[dict] = deque(maxlen=self.window)
        # Karate: timestamped frames, window = last realtime_window_s seconds.
        self.window_s = s.realtime_window_s
        self.timed: deque[tuple[float, dict]] = deque(maxlen=400)
        self.tracker = LiveTracker()
        self.selector = PlayerSelector(rank_by="limbs" if self.sport == "karate" else "motion")
        self._jpeg_every = s.realtime_jpeg_every
        self.frame_hw: tuple[int, int] | None = None
        self.state: dict = {
            "running": False,
            "frames_processed": 0,
            "fps": 0.0,
            "people_now": 0,
            "latest_keypoints": None,
            "latest_frame_jpeg": None,
            "prediction": None,
            "inference_s": None,
            "buffer_fill": 0,
            "window": self.window,
            "error": None,
            "openpose_device": None,
            "mmaction_device": None,
            "frame_size": None,
            "sport": self.sport,
            # "warming" | "no_person" | "ok" - lets the UI say WHY there is no label.
            "status": "warming",
            # Keypoints of exactly the frame in latest_frame_jpeg, so the overlay
            # is drawn on the image it was detected in (latest_keypoints can be a
            # frame or two ahead of the last shipped JPEG).
            "frame_keypoints": None,
            # Camera-to-processed delay: OpenPose writes each frame's JSON the
            # moment it is done, so now - file mtime is the real pipeline lag.
            "lag_ms": None,
            # Share of the 17 COCO joints found (median over the window).
            "body_coverage": None,
            "window_seconds": s.realtime_window_s,
        }

    # ---------------------------------------------------------------- start --
    def start(self) -> None:
        if self.engine in ("rtmpose", "yolo11"):
            self._start_python_engine()
        else:
            self._start_openpose()
        # Persistent recognizer so each prediction costs ~0.1s, not ~5s.
        self._start_mmaction()

        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        with self._lock:
            self.state["running"] = True
        self.log.info(f"live session started ({self.engine})", stage="live")

    def _start_python_engine(self) -> None:
        """RTMPose / YOLO11 camera loop (backend/pose_live.py) in the MMAction2 venv."""
        s = get_settings()
        if not s.pose_engine_ready(self.engine):
            raise RealtimeError(f"{self.engine} is not installed")
        cfg = {
            "engine": self.engine,
            "libs": [str(s.pose_libs_dir), str(s.yolo_libs_dir)],
            "models": s.pose_engine_models(self.engine),
            "device": s.device,
            "camera_index": s.realtime_camera_index,
            "resolution": s.realtime_camera_resolution,
            "json_dir": str(self.json_dir),
            "img_dir": str(self.img_dir),
            "max_people": s.realtime_max_people,
            "reid": s.reid_cfg,
        }
        self.log.info(f"live {self.engine} camera loop starting", stage="live")
        self._op = subprocess.Popen(
            [str(s.active_mmaction_python), str(BACKEND_DIR / "pose_live.py"), json.dumps(cfg)],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.DEVNULL,  # progress not needed; avoids a full pipe blocking it
            stderr=subprocess.PIPE,
            text=True,
        )
        labels = {"rtmpose": "RTMPose-m + RTMDet-m", "yolo11": "YOLO11m-pose"}
        self.state["openpose_device"] = f"{labels[self.engine]} on {s.device}"

    def _start_openpose(self) -> None:
        s = get_settings()
        bin_path = s.realtime_openpose_bin
        if not bin_path.exists():
            raise RealtimeError(f"OpenPose binary not found: {bin_path}")

        model_folder = str(s.openpose_model_dir).replace("\\", "/").rstrip("/") + "/"
        cmd = [
            str(bin_path),
            "--camera", str(s.realtime_camera_index),
            "--camera_resolution", s.realtime_camera_resolution,
            "--model_folder", model_folder,
            "--model_pose", "BODY_25",
            "--net_resolution", s.realtime_net_resolution,
            "--write_json", str(self.json_dir),
            "--write_images", str(self.img_dir),
            "--write_images_format", "jpg",
            "--display", "0",
            "--render_pose", str(s.realtime_render_pose),
            "--number_people_max", str(s.realtime_max_people),
        ]
        self.log.info("live OpenPose command: " + " ".join(cmd), stage="live")
        self._op = subprocess.Popen(
            cmd,
            cwd=str(bin_path.parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.state["openpose_device"] = s.realtime_device_label

    def _model_cfg(self) -> dict:
        s = get_settings()
        return {
            "config": str(s.mmaction_config),
            "checkpoint": str(s.checkpoint_path),
            "label_map": str(s.label_map_path),
        }

    def _start_mmaction(self) -> None:
        s = get_settings()
        cfg = {**self._model_cfg(), "device": s.device, "num_clips": s.realtime_num_clips}
        self._mm = subprocess.Popen(
            [str(s.active_mmaction_python), str(WORKER), json.dumps(cfg)],
            cwd=str(s.mmaction2_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_msg(timeout=180)
        if not ready or not ready.get("ready"):
            raise RealtimeError(f"MMAction2 worker failed to start: {ready}")
        self.state["mmaction_device"] = ready.get("device_used")
        self.log.info(
            f"MMAction2 worker ready on {ready.get('device_used')} "
            f"(num_clips={ready.get('num_clips')})",
            stage="live",
        )

    def _read_msg(self, timeout: float = 30.0) -> dict | None:
        """Read one protocol line, skipping mmengine's stdout chatter."""
        assert self._mm is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._mm.stdout.readline()
            if not line:
                return None
            if line.startswith(MSG_PREFIX):
                return json.loads(line[len(MSG_PREFIX):])
        return None

    # ---------------------------------------------------------------- watch --
    def _watch(self) -> None:
        seen: set[str] = set()
        times: deque[float] = deque(maxlen=30)
        frame_no = 0
        try:
            while not self._stop.is_set():
                if self._op and self._op.poll() is not None:
                    err = (self._op.stderr.read() or "")[-1500:] if self._op.stderr else ""
                    with self._lock:
                        self.state["error"] = f"OpenPose exited ({self._op.returncode}). {err}"
                        self.state["running"] = False
                    self.log.error(f"live OpenPose exited: {err}", stage="live")
                    self.stop()  # also end the MMAction2 worker
                    return

                files = sorted(self.json_dir.glob("*_keypoints.json"))
                new = [f for f in files if f.name not in seen]
                if not new:
                    time.sleep(0.02)
                    continue

                # If processing ever falls behind, jump to the newest frames
                # instead of working through a backlog - that backlog is what
                # makes a "live" view drift seconds behind the camera.
                if len(new) > 6:
                    for old in new[:-3]:
                        seen.add(old.name)
                    new = new[-3:]
                for fp in new:
                    seen.add(fp.name)
                    try:
                        data = json.loads(fp.read_text(encoding="utf-8"))
                        lag = (time.time() - fp.stat().st_mtime) * 1000
                    except (OSError, json.JSONDecodeError):
                        continue  # still being written; skip this tick
                    with self._lock:
                        self.state["lag_ms"] = int(lag)
                    frame_no += 1
                    times.append(time.time())
                    try:
                        self._ingest(data, frame_no, fp.name)
                    except Exception as exc:  # noqa: BLE001
                        # One bad frame must not end the session (an
                        # IndexError here used to kill the watcher thread and
                        # freeze the live view). Report it and keep going.
                        with self._lock:
                            self.state["error"] = f"frame {frame_no}: {type(exc).__name__}: {exc}"
                        self.log.error(f"live frame {frame_no} failed: {exc!r}", stage="live")

                self._prune(seen)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
                self.state["running"] = False
            self.log.error(f"live watcher crashed: {exc}", stage="live")
            # Nothing reads frames any more - release the camera and GPU now
            # instead of leaving OpenPose running until someone clicks Stop.
            self.stop()

    def _ingest(self, data: dict, frame_no: int, json_name: str = "") -> None:
        people = _parse_people(data.get("people", []))
        if any(p["engine_id"] is not None for p in people):
            # YOLO11 already tracked them (ByteTrack) - far steadier identities
            # than nearest-centroid when people cross or occlude each other.
            for p in people:
                p["track_id"] = p["engine_id"]
        else:
            self.tracker.update(people)
        # Players = present AND moving (same rule as batch); they are both the
        # skeletons sent to the classifier and the ones labelled on screen.
        if self.frame_hw:
            self.selector.set_frame(*self.frame_hw)
        self.selector.update(people)
        by_tid = {p["track_id"]: p for p in people if p.get("track_id") is not None}
        present = {tid: p["centroid"] for tid, p in by_tid.items() if p["centroid"]}
        # Identity signature: OSNet re-id when available, else clothing colours.
        reids = {tid: p["reid"] for tid, p in by_tid.items() if p.get("reid") is not None}
        if reids:
            slots = self.selector.players(self.max_persons, present, reids, key="reid")
        else:
            apps = {tid: p.get("appearance") for tid, p in by_tid.items() if p.get("appearance")}
            slots = self.selector.players(self.max_persons, present, apps)
        players = [
            {
                "label": f"Player {i + 1}",
                "slot": i,
                "keypoints": [[round(v, 1) for v in kp] for kp in by_tid[tid]["keypoints"]],
                "bbox": by_tid[tid].get("bbox"),
            }
            for i, tid in enumerate(slots)
            if tid is not None and tid in by_tid
        ]

        kp_frame, sc_frame = [], []
        for tid in slots:
            person = by_tid.get(tid)
            if person is None:
                kp_frame.append([[0.0, 0.0]] * 17)
                sc_frame.append([0.0] * 17)
            else:
                xy, sc = _to_coco17(person["keypoints"])
                kp_frame.append(xy)
                sc_frame.append(sc)
        while len(kp_frame) < 1:  # always keep at least one slot
            kp_frame.append([[0.0, 0.0]] * 17)
            sc_frame.append([0.0] * 17)

        entry = {"keypoints": kp_frame, "keypoint_scores": sc_frame}

        jpeg = self._frame_jpeg(json_name) if frame_no % self._jpeg_every == 0 else None
        kps = [[[round(v, 1) for v in kp] for kp in p["keypoints"]] for p in people[:8]]
        with self._lock:
            self.state["frames_processed"] = frame_no
            self.state["people_now"] = len(people)
            self.state["latest_keypoints"] = kps
            self.state["latest_players"] = players
            if jpeg:
                self.state["latest_frame_jpeg"] = jpeg
                self.state["frame_keypoints"] = kps
                self.state["frame_players"] = players  # paired with this exact image

        self._on_frame(entry, frame_no)

    def _on_frame(self, entry: dict, frame_no: int) -> None:
        """Karate: keep the last window_s SECONDS and classify when due.

        Overridden by the swimming session, which has its own timing.
        """
        now = time.time()
        self.timed.append((now, entry))
        while self.timed and now - self.timed[0][0] > self.window_s:
            self.timed.popleft()
        span = now - self.timed[0][0]
        self.buffer = deque((e for _, e in self.timed), maxlen=len(self.timed))
        with self._lock:
            self.state["buffer_fill"] = min(int(span / self.window_s * 100), 100)
            self.state["window"] = 100
        if span >= 0.9 * self.window_s and frame_no % self.infer_every == 0:
            present = sum(person_present(e) for e in self.buffer) / len(self.buffer)
            counts = sorted(
                sum(1 for c in e["keypoint_scores"][0] if c > 0.05) for e in self.buffer
            )
            with self._lock:
                self.state["body_coverage"] = round(counts[len(counts) // 2] / 17, 2)
            if present < MIN_PRESENT_SHARE:
                with self._lock:
                    self.state["status"] = "no_person"
                    self.state["prediction"] = None
                return
            self._infer()

    def _frame_jpeg(self, json_name: str) -> str | None:
        """JPEG of the SAME frame as json_name; falls back to the newest one."""
        if json_name.endswith("_keypoints.json"):
            img = self.img_dir / json_name.replace("_keypoints.json", "_rendered.jpg")
            try:
                raw = img.read_bytes()
                if raw:
                    self.state["frame_size"] = len(raw)
                    if self.frame_hw is None:
                        dims = jpeg_dimensions(raw)
                        if dims:
                            self.frame_hw = dims
                            self.state["frame_hw"] = list(dims)
                    return base64.b64encode(raw).decode("ascii")
            except OSError:
                pass  # not written yet - OpenPose writes JSON and JPEG separately
        return self._latest_jpeg()

    def _latest_jpeg(self) -> str | None:
        try:
            imgs = sorted(self.img_dir.glob("*.jpg"))
            if not imgs:
                return None
            newest = imgs[-1]
            raw = newest.read_bytes()
            if not raw:
                return None
            self.state["frame_size"] = len(raw)
            if self.frame_hw is None:
                dims = jpeg_dimensions(raw)
                if dims:
                    self.frame_hw = dims
                    self.state["frame_hw"] = list(dims)
            return base64.b64encode(raw).decode("ascii")
        except OSError:
            return None

    def _infer(self) -> None:
        if not self._mm or self._mm.poll() is not None:
            return
        h, w = self.frame_hw or get_settings().realtime_frame_hw
        self._req_id += 1
        req = {
            "id": self._req_id,
            # Karate/NTU60 live only: fill holes so missing joints are not
            # read as sitting in the frame's top-left corner.
            "pose_results": impute_missing(_single_or_pair(list(self.buffer))),
            "img_shape": [h, w],
            "topk": 5,
        }
        try:
            self._mm.stdin.write(json.dumps(req) + "\n")
            self._mm.stdin.flush()
            resp = self._read_msg(timeout=60)
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self.state["error"] = f"MMAction2 worker pipe closed: {exc}"
            return
        if not resp:
            return
        with self._lock:
            if resp.get("ok"):
                self.state["prediction"] = {
                    "top1": resp.get("top1"),
                    "topk": resp.get("topk"),
                    "num_clips": resp.get("num_clips"),
                }
                self.state["inference_s"] = resp.get("inference_s")
                self.state["status"] = "ok"
            else:
                self.state["error"] = resp.get("error")

    def _prune(self, seen: set[str], keep: int = 40) -> None:
        """Bound disk usage - a live session writes a file per frame forever."""
        for d, pat in ((self.json_dir, "*_keypoints.json"), (self.img_dir, "*.jpg")):
            files = sorted(d.glob(pat))
            for old in files[:-keep]:
                try:
                    old.unlink()
                except OSError:
                    pass

    # ----------------------------------------------------------------- read --
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def stop(self) -> None:
        self._stop.set()
        if self.engine != "openpose" and self._op and self._op.poll() is None:
            # A venv python.exe launches the real interpreter as a CHILD; killing
            # only the launcher would leave the child holding the webcam.
            subprocess.run(["taskkill", "/PID", str(self._op.pid), "/T", "/F"],
                           capture_output=True, timeout=20)
        for proc, name in ((self._op, "openpose"), (self._mm, "mmaction")):
            if proc and proc.poll() is None:
                try:
                    if name == "mmaction" and proc.stdin:
                        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    proc.terminate()
                    proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                    except OSError:
                        pass
        with self._lock:
            self.state["running"] = False
        self.log.info("live session stopped", stage="live")


def kill_orphaned_openpose() -> int:
    """Kill stray OpenPoseDemo.exe processes left over from a previous run.

    A live session's OpenPose is a child of the backend. If the backend is
    killed or restarted while a session is running, that child is orphaned and
    keeps the camera AND the GPU. This is not theoretical: an orphan held
    ~2 GB of the 4 GB VRAM and slowed a batch job from 0.071 to 0.782 s/frame -
    an 11x regression that looked like a GPU problem.

    Only processes whose image path is inside a configured OpenPose directory
    are touched, so an unrelated OpenPose elsewhere on the machine is left alone.
    Returns how many were killed.
    """
    s = get_settings()
    roots = {str(s.openpose_dir).lower(), str(s.openpose_gpu_dir).lower()}
    killed = 0
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='OpenPoseDemo.exe'",
             "get", "ProcessId,ExecutablePath", "/format:csv"],
            capture_output=True, text=True, timeout=30,
        )
        for line in (out.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[-1].isdigit():
                continue
            exe_path, pid = parts[-2].lower(), parts[-1]
            if not any(exe_path.startswith(r) for r in roots if r):
                continue
            subprocess.run(["taskkill", "/PID", pid, "/F"],
                           capture_output=True, timeout=20)
            killed += 1
    except (OSError, subprocess.SubprocessError):
        return killed
    return killed


def kill_live_camera_openpose() -> int:
    """Kill every OpenPoseDemo.exe that owns a camera (``--camera`` in argv).

    Batch jobs run OpenPose with ``--video`` and are never touched. Live ones
    must not survive their session: an orphan keeps the webcam (so the next
    session gets no frames) and ~2 GB of VRAM. Seen for real: after the watcher
    crashed, Start created a new session over the dead one and its OpenPose
    kept running after Stop.
    """
    killed = 0
    # OpenPose with --camera, and the RTMPose/YOLO11 camera loop (pose_live.py).
    for image, marker in (("OpenPoseDemo.exe", "--camera"), ("python.exe", "pose_live.py")):
        try:
            out = subprocess.run(
                ["wmic", "process", "where", f"name='{image}'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=30,
            )
            for line in (out.stdout or "").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3 or not parts[-1].isdigit() or marker not in line:
                    continue
                subprocess.run(["taskkill", "/PID", parts[-1], "/F", "/T"],
                               capture_output=True, timeout=20)
                killed += 1
        except (OSError, subprocess.SubprocessError):
            pass
    return killed


_session: RealtimeSession | None = None
_session_lock = threading.Lock()
# `running` is only set at the END of start(), which takes ~20s (CUDA init plus
# the ST-GCN load). Without a separate "starting" latch, a second click during
# that window passes the running check and spawns a SECOND OpenPose process that
# fights the first one for the camera.
_starting = False


class SessionStarting(RuntimeError):
    """A start is already in progress; the caller should just wait."""


def get_session() -> RealtimeSession | None:
    with _session_lock:
        return _session


def is_starting() -> bool:
    with _session_lock:
        return _starting


def start_session(sport: str = "karate", options: dict | None = None) -> RealtimeSession:
    global _session, _starting
    with _session_lock:
        if _session is not None and _session.snapshot().get("running"):
            if _session.sport != sport:
                raise SessionStarting(
                    f"A {_session.sport} live session is running; stop it first "
                    "(there is one camera and one GPU)."
                )
            return _session
        if _starting:
            raise SessionStarting(
                "A live session is already starting; wait for it to come up."
            )
        _starting = True
    try:
        return _do_start(sport, options or {})
    finally:
        with _session_lock:
            _starting = False


def _do_start(sport: str, options: dict) -> RealtimeSession:
    """Spawn the session. Runs outside the lock: start() blocks for ~20s and
    holding the lock would stall every status poll for that whole window."""
    global _session
    s = get_settings()
    # A crashed or stopped-but-not-cleaned session must release the camera and
    # GPU before a new one starts, or the new OpenPose cannot open the webcam.
    with _session_lock:
        old, _session = _session, None
    if old is not None:
        old.stop()
    kill_live_camera_openpose()
    d = s.data_dir / "live"
    # Fresh directory each run so stale frames cannot be replayed as new ones.
    if d.exists():
        for sub in ("json", "frames"):
            for f in (d / sub).glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
    log = JobLogger("live", d)
    engine = options.pop("engine", "openpose")
    if sport == "swimming":
        from .swim_live_service import SwimRealtimeSession  # swimming-only logic
        session: RealtimeSession = SwimRealtimeSession(d, log, engine=engine, **options)
    else:
        session = RealtimeSession(d, log, engine=engine)
    try:
        session.start()
    except Exception:
        # A half-started session (OpenPose up, worker failed) must not keep the
        # camera and GPU - that is how orphans slowed everything down before.
        session.stop()
        raise
    with _session_lock:
        _session = session
    return session


def stop_session() -> None:
    """Stop the live session, whatever state it is in, and free the camera.

    The session is detached under the lock but stopped OUTSIDE it: stop() waits
    for processes to exit, and holding the lock meanwhile would block the
    WebSocket loop (which reads the session) and freeze every open page.
    """
    global _session
    with _session_lock:
        old, _session = _session, None
    if old is not None:
        old.stop()
    kill_live_camera_openpose()
