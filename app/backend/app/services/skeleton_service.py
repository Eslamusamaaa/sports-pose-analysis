"""Parses OpenPose BODY_25 JSON into a structured, tracked skeleton sequence.

Real OpenPose JSON format, verified against v1.7.0 output rather than assumed:

    {
      "version": 1.3,
      "people": [
        {
          "person_id": [-1],
          "pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ...],   # 75 floats = 25*3
          "face_keypoints_2d": [], "hand_left_keypoints_2d": [], ...
        },
        ...
      ]
    }

Two things matter downstream:

1. ``pose_keypoints_2d`` is a FLAT list of 75 numbers: 25 joints x (x, y, confidence),
   in pixel coordinates of the input frame. Undetected joints are exactly (0, 0, 0).

2. ``person_id`` is ``[-1]`` unless OpenPose runs with ``--tracking``, i.e. there is
   NO identity association across frames. People also arrive in an arbitrary order
   that changes frame to frame. Feeding that straight into a GCN would scramble
   each person slot over time, so this module runs a small greedy centroid tracker
   to build temporally consistent tracks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..utils.logging_utils import JobLogger
from .keypoints import BODY_25_NAMES

NUM_BODY25 = 25
VALUES_PER_JOINT = 3  # x, y, confidence


class SkeletonError(RuntimeError):
    pass


def _parse_person(raw: dict) -> list[tuple[float, float, float]] | None:
    flat = raw.get("pose_keypoints_2d") or []
    if len(flat) != NUM_BODY25 * VALUES_PER_JOINT:
        # A person with face/hand-only data, or an unexpected layout. Skip it
        # rather than silently reshaping something that is not BODY_25.
        return None
    return [
        (float(flat[i * 3]), float(flat[i * 3 + 1]), float(flat[i * 3 + 2]))
        for i in range(NUM_BODY25)
    ]


def _centroid(joints: list[tuple[float, float, float]], min_conf: float = 0.1):
    pts = [(x, y) for x, y, c in joints if c > min_conf and (x or y)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# BODY_25 indices for limb activity.
_NECK, _MIDHIP, _RHIP, _LHIP = 1, 8, 9, 12
_LIMB_ENDS = (4, 7, 11, 14)  # RWrist, LWrist, RAnkle, LAnkle
_LIMB_C = 0.1


def _body_center_and_scale(kp) -> tuple[tuple[float, float], float] | None:
    """Hip centre and torso length (neck -> hips) of one skeleton, or None."""
    if kp[_MIDHIP][2] > _LIMB_C:
        hip = (kp[_MIDHIP][0], kp[_MIDHIP][1])
    elif kp[_RHIP][2] > _LIMB_C and kp[_LHIP][2] > _LIMB_C:
        hip = ((kp[_RHIP][0] + kp[_LHIP][0]) / 2, (kp[_RHIP][1] + kp[_LHIP][1]) / 2)
    else:
        return None
    if kp[_NECK][2] <= _LIMB_C:
        return None
    torso = math.dist(hip, (kp[_NECK][0], kp[_NECK][1]))
    return (hip, torso) if torso > 1 else None


def limb_step(prev, cur) -> float | None:
    """How much the hands and feet moved RELATIVE TO THE BODY between two
    consecutive frames of one person, in torso lengths.

    Relative to the hips, so walking across the mat (whole body translating)
    scores low while punching / kicking / bouncing in stance scores high; in
    torso lengths, so a fighter near the camera and one far away are measured
    on the same scale. General for any video - no pixel or frame constants.
    """
    a, b = _body_center_and_scale(prev), _body_center_and_scale(cur)
    if a is None or b is None:
        return None
    (ha, ta), (hb, tb) = a, b
    torso = (ta + tb) / 2
    moves = []
    for j in _LIMB_ENDS:
        if prev[j][2] > _LIMB_C and cur[j][2] > _LIMB_C:
            ra = (prev[j][0] - ha[0], prev[j][1] - ha[1])
            rb = (cur[j][0] - hb[0], cur[j][1] - hb[1])
            moves.append(math.dist(ra, rb) / torso)
    return sum(moves) / len(moves) if moves else None


def _mean_conf(joints: list[tuple[float, float, float]]) -> float:
    confs = [c for _, _, c in joints if c > 0]
    return sum(confs) / len(confs) if confs else 0.0


def _valid_count(joints: list[tuple[float, float, float]], min_conf: float = 0.1) -> int:
    return sum(1 for _, _, c in joints if c > min_conf)


def parse_openpose_dir(json_dir: Path, log: JobLogger) -> dict:
    """Read every *_keypoints.json in frame order and parse BODY_25 people."""
    files = sorted(json_dir.glob("*_keypoints.json"))
    if not files:
        raise SkeletonError(f"No OpenPose JSON files found in {json_dir}")

    frames: list[dict] = []
    versions: set[str] = set()
    total_people = 0
    skipped = 0

    for frame_idx, fp in enumerate(files):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkeletonError(f"Could not read {fp.name}: {exc}") from exc

        if "version" in data:
            versions.add(str(data["version"]))

        people_out = []
        for raw in data.get("people", []):
            joints = _parse_person(raw)
            if joints is None:
                skipped += 1
                continue
            pid = raw.get("person_id")
            people_out.append(
                {
                    # OpenPose always writes -1; YOLO11 writes its ByteTrack id.
                    "openpose_person_id": (pid[0] if isinstance(pid, list) and pid else -1),
                    # Detector box [x1, y1, x2, y2] (RTMDet / YOLO11 only).
                    "bbox": raw.get("bbox"),
                    "bbox_score": raw.get("bbox_score"),
                    # Clothing colour signature (RTMPose / YOLO11 engines).
                    "appearance": raw.get("appearance"),
                    "reid_idx": raw.get("reid_idx"),
                    "keypoints": joints,
                    "mean_confidence": round(_mean_conf(joints), 4),
                    "valid_keypoints": _valid_count(joints),
                    "centroid": _centroid(joints),
                }
            )
        total_people += len(people_out)
        frames.append({"frame": frame_idx, "source_file": fp.name, "people": people_out})

    if skipped:
        log.warning(f"skipped {skipped} person entries without 75-value BODY_25 data",
                    stage="skeleton")

    # OSNet re-id embeddings (one row per detected person) written next to the
    # JSON folder by pose_extract.py; attach each person's 512-d vector.
    reid_file = json_dir.parent / "reid.npy"
    if reid_file.exists():
        import numpy as np

        mat = np.load(reid_file).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        n_att = 0
        for f in frames:
            for p in f["people"]:
                i = p.get("reid_idx")
                if i is not None and 0 <= i < len(mat):
                    p["reid"] = mat[i]
                    n_att += 1
        log.info(f"re-id: {n_att} person embeddings loaded (OSNet-AIN, 512-d)", stage="skeleton")

    log.info(
        f"parsed {len(frames)} frames, {total_people} person detections "
        f"(avg {total_people / len(frames):.2f} people/frame), "
        f"OpenPose JSON version(s): {', '.join(sorted(versions)) or 'unknown'}",
        stage="skeleton",
    )
    return {
        "frames": frames,
        "frame_count": len(frames),
        "total_person_detections": total_people,
        "avg_people_per_frame": round(total_people / len(frames), 3) if frames else 0.0,
        "max_people_in_a_frame": max((len(f["people"]) for f in frames), default=0),
        "json_versions": sorted(versions),
        "keypoint_format": "BODY_25",
        "num_keypoints": NUM_BODY25,
        "keypoint_names": list(BODY_25_NAMES),
        "coordinate_space": "pixels of the input frame; (0,0,0) means undetected",
    }


def track_people(
    parsed: dict,
    log: JobLogger,
    max_distance_ratio: float = 0.2,
    motion_weight_floor: float = 0.25,
    rank_by: str = "motion",
) -> dict:
    """Assign temporally consistent track ids via greedy centroid matching.

    OpenPose emits no cross-frame identity (person_id is always -1), so without
    this the Nth person in frame t is unrelated to the Nth person in frame t+1.

    This is intentionally simple - nearest-centroid, gated by a distance
    threshold relative to frame diagonal. It is adequate for a PoC with one
    dominant subject but is NOT a real multi-object tracker: it has no
    re-identification and no occlusion handling, so long gaps start a new track.
    """
    frames = parsed["frames"]
    # Estimate a frame scale from observed coordinates to normalise distances.
    xs = [x for f in frames for p in f["people"] for x, _, c in p["keypoints"] if c > 0.1]
    ys = [y for f in frames for p in f["people"] for _, y, c in p["keypoints"] if c > 0.1]
    if xs and ys:
        scale = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    else:
        scale = 1.0
    gate = scale * max_distance_ratio

    next_track_id = 0
    active: dict[int, tuple[float, float]] = {}  # track_id -> last centroid
    track_stats: dict[int, dict] = {}

    # If the pose engine already tracked people (YOLO11 + ByteTrack writes its
    # id into person_id), use those ids: ByteTrack matches boxes with motion
    # prediction and keeps identities through short occlusions, which the
    # nearest-centroid matcher below cannot. Detections ByteTrack left without
    # an id (low-confidence, not yet confirmed) stay untracked.
    dets = [p for f in frames for p in f["people"]]
    engine_ids = bool(dets) and sum(p["openpose_person_id"] >= 0 for p in dets) >= 0.5 * len(dets)
    if engine_ids:
        log.info("tracking: using the pose engine's own track ids (ByteTrack)", stage="skeleton")
    elif dets and rank_by == "limbs":
        # Karate with an engine that gives no ids (OpenPose, RTMPose): the
        # nearest-centroid matcher below cannot cope with a crowded hall - on a
        # club video with ~8 people per frame it made 22 "tracks" that each
        # jumped between people for the whole clip. Run ByteTrack (Kalman
        # motion + box overlap) instead and use its ids exactly like YOLO's.
        from . import bytetrack

        bytetrack.track_sequence(frames, None)
        for p in dets:
            p["openpose_person_id"] = p["track_id"] if p.get("track_id") is not None else -1
        engine_ids = True
        log.info("tracking: ByteTrack (motion + box overlap) - the pose engine gives no ids",
                 stage="skeleton")

    for frame in frames:
        assigned: dict[int, int] = {}  # person index -> track id
        if engine_ids:
            assigned = {
                i: p["openpose_person_id"]
                for i, p in enumerate(frame["people"])
                if p["openpose_person_id"] >= 0
            }
            candidates = []
        else:
            candidates = [
                (i, p["centroid"]) for i, p in enumerate(frame["people"]) if p["centroid"]
            ]
        used_tracks: set[int] = set()
        # Greedy: sort all (person, track) pairs by distance and take best first.
        pairs = []
        for i, cen in candidates:
            for tid, last in active.items():
                pairs.append((math.dist(cen, last), i, tid))
        pairs.sort()
        for dist, i, tid in pairs:
            if i in assigned or tid in used_tracks or dist > gate:
                continue
            assigned[i] = tid
            used_tracks.add(tid)

        for i, cen in candidates:
            if i not in assigned:
                assigned[i] = next_track_id
                next_track_id += 1

        for i, person in enumerate(frame["people"]):
            tid = assigned.get(i)
            if tid is None:
                person["track_id"] = None
                continue
            person["track_id"] = tid
            if person["centroid"]:
                active[tid] = person["centroid"]
            st = track_stats.setdefault(
                tid,
                {
                    "track_id": tid, "frames": 0, "conf_sum": 0.0, "valid_sum": 0,
                    "motion_sum": 0.0, "prev": None,
                    "limb_sum": 0.0, "limb_n": 0, "prev_kp": None, "prev_frame": None,
                    "x_sum": 0.0, "y_sum": 0.0,
                },
            )
            st["frames"] += 1
            st["conf_sum"] += person["mean_confidence"]
            if st["prev_kp"] is not None and st["prev_frame"] == frame["frame"] - 1:
                ls = limb_step(st["prev_kp"], person["keypoints"])
                if ls is not None:
                    st["limb_sum"] += ls
                    st["limb_n"] += 1
            st["prev_kp"], st["prev_frame"] = person["keypoints"], frame["frame"]
            st["valid_sum"] += person["valid_keypoints"]
            cen = person["centroid"]
            if cen:
                st["x_sum"] += cen[0]
                st["y_sum"] += cen[1]
                # Accumulate how far this person actually travels. This is what
                # separates an athlete from a stationary referee or spectator.
                if st["prev"] is not None:
                    st["motion_sum"] += math.dist(cen, st["prev"])
                st["prev"] = cen

    tracks = []
    # Normalise motion against the busiest track so the weighting is scale-free
    # (it must behave the same for a 480p clip and a 4K one).
    max_motion = max(
        (st["motion_sum"] / max(st["frames"], 1) for st in track_stats.values()),
        default=0.0,
    ) or 1.0
    max_limb = max(
        (st["limb_sum"] / st["limb_n"] for st in track_stats.values() if st["limb_n"]),
        default=0.0,
    ) or 1.0

    for st in track_stats.values():
        n = max(st["frames"], 1)
        motion_per_frame = st["motion_sum"] / n
        # WHY MOTION IS IN THE SCORE
        # -------------------------
        # The original ranking was confidence x presence, which systematically
        # picked the WRONG people in sports footage: a referee or spectator
        # stands still, never leaves frame and is never motion-blurred, so they
        # score higher than the athletes - who move fast, blur, occlude each
        # other and briefly exit frame. Measured on a real karate match, the two
        # selected tracks moved 5.6 and 11.0 px/frame while the actual fighters
        # moved 16-21 px/frame; the model ended up classifying the officials.
        #
        # Motion is therefore a first-class term, not a tiebreaker.
        motion_factor = motion_per_frame / max_motion
        limb_activity = st["limb_sum"] / st["limb_n"] if st["limb_n"] else 0.0
        if rank_by == "limbs":
            # Karate: rank by hands/feet movement relative to the body, so a
            # coach walking around the mat does not outrank the fighters.
            motion_factor = limb_activity / max_limb
        tracks.append(
            {
                "track_id": st["track_id"],
                "frames_present": st["frames"],
                "mean_confidence": round(st["conf_sum"] / n, 4),
                "mean_valid_keypoints": round(st["valid_sum"] / n, 2),
                "motion_per_frame": round(motion_per_frame, 2),
                "limb_activity": round(limb_activity, 4),
                "mean_x": round(st["x_sum"] / n, 1),
                "mean_y": round(st["y_sum"] / n, 1),
                "score": round(
                    (st["conf_sum"] / n)
                    * st["frames"]
                    * (motion_weight_floor + (1 - motion_weight_floor) * motion_factor),
                    4,
                ),
            }
        )
    tracks.sort(key=lambda t: t["score"], reverse=True)

    log.info(
        f"tracking: {len(tracks)} tracks over {parsed['frame_count']} frames "
        f"(gate={gate:.1f}px, motion-weighted ranking)",
        stage="skeleton",
    )
    for t in tracks[:4]:
        log.info(
            f"  track {t['track_id']}: {t['frames_present']} frames, "
            f"conf {t['mean_confidence']}, motion {t['motion_per_frame']}px/frame, "
            f"score {t['score']}",
            stage="skeleton",
        )
    parsed["tracks"] = tracks
    parsed["track_count"] = len(tracks)
    parsed["tracker"] = "ByteTrack (YOLO11)" if engine_ids else "nearest-centroid"
    return parsed


def link_player_tracklets(
    parsed: dict,
    n_players: int,
    log: JobLogger,
    max_gap_s: float = 2.0,
    max_dist_ratio: float = 0.25,
    fps: float = 30.0,
) -> dict:
    """Join each player's track pieces so the player is labelled from frame 0.

    Even ByteTrack loses a person who leaves the frame or is hidden for a moment
    and gives them a NEW id when they come back. Measured on a real kumite clip:
    Player 2 walked out at the left edge (track 2, frames 0-87) and came back at
    the same edge as track 29 (frame 108) - so the first 3.6 s had no "Player 2";
    Player 1's track ended at frame 550 and continued as track 85 at frame 552
    in the same spot.

    For each player (the top-ranked tracks), repeatedly absorb the best piece
    that ends shortly BEFORE the player's first frame (or starts shortly AFTER
    its last) without overlapping it in time, and whose end/start position is
    close. "Best" = smallest normalised time gap + distance. Other players'
    pieces are never taken.
    """
    tracks = parsed.get("tracks") or []
    if not tracks or n_players < 1:
        return parsed
    # Seconds -> frames for THIS video, so 25/30/60 fps all allow the same 2 s.
    max_gap_frames = max(1, round(max_gap_s * (fps or 30.0)))
    players = [t["track_id"] for t in tracks[:n_players]]

    span: dict[int, dict] = {}
    cents = []
    for f in parsed["frames"]:
        for p in f["people"]:
            tid, cen = p.get("track_id"), p["centroid"]
            if tid is None or cen is None:
                continue
            cents.append(cen)
            s = span.setdefault(tid, {"first": f["frame"], "last": f["frame"], "c0": cen, "c1": cen})
            s["last"], s["c1"] = f["frame"], cen
    if not cents:
        return parsed
    diag = math.hypot(max(c[0] for c in cents) - min(c[0] for c in cents),
                      max(c[1] for c in cents) - min(c[1] for c in cents)) or 1.0
    max_dist = max_dist_ratio * diag
    owner: dict[int, int] = {}  # absorbed piece -> player

    for pid in players:
        for direction in ("back", "fwd"):
            while True:
                ps = span[pid]
                best, best_cost = None, None
                for tid, s in span.items():
                    if tid in players or tid in owner:
                        continue
                    if direction == "back":
                        gap = ps["first"] - s["last"]
                        dist = math.dist(s["c1"], ps["c0"])
                    else:
                        gap = s["first"] - ps["last"]
                        dist = math.dist(ps["c1"], s["c0"])
                    if gap < 1 or gap > max_gap_frames or dist > max_dist:
                        continue  # overlaps in time, too far in time, or too far away
                    cost = gap / max_gap_frames + dist / max_dist
                    if best_cost is None or cost < best_cost:
                        best, best_cost = tid, cost
                if best is None:
                    break
                owner[best] = pid
                s = span[best]
                if direction == "back":
                    ps["first"], ps["c0"] = s["first"], s["c0"]
                else:
                    ps["last"], ps["c1"] = s["last"], s["c1"]
                log.info(f"players: track {best} (frames {s['first']}-{s['last']}) joined "
                         f"track {pid} ({'before' if direction == 'back' else 'after'} it)",
                         stage="skeleton")

    if not owner:
        return parsed
    for f in parsed["frames"]:
        for p in f["people"]:
            if p.get("track_id") in owner:
                p["track_id"] = owner[p["track_id"]]
    present = {pid: sum(1 for f in parsed["frames"] if any(p.get("track_id") == pid for p in f["people"]))
               for pid in players}
    by_id = {t["track_id"]: t for t in tracks}
    head = []
    for pid in players:
        t = dict(by_id[pid])
        t["frames_present"] = present[pid]
        t["joined_tracks"] = sorted(k for k, v in owner.items() if v == pid)
        head.append(t)
    parsed["tracks"] = head + [t for t in tracks if t["track_id"] not in players and t["track_id"] not in owner]
    parsed["track_count"] = len(parsed["tracks"])
    return parsed


def save_skeleton(parsed: dict, dest: Path, log: JobLogger) -> None:
    """Write the parsed skeleton. The raw OpenPose JSON is left untouched."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keypoint_format": parsed["keypoint_format"],
        "num_keypoints": parsed["num_keypoints"],
        "keypoint_names": parsed["keypoint_names"],
        "coordinate_space": parsed["coordinate_space"],
        "frame_count": parsed["frame_count"],
        "total_person_detections": parsed["total_person_detections"],
        "avg_people_per_frame": parsed["avg_people_per_frame"],
        "max_people_in_a_frame": parsed["max_people_in_a_frame"],
        "json_versions": parsed["json_versions"],
        "tracks": parsed.get("tracks", []),
        "frames": [
            {
                "frame": f["frame"],
                "source_file": f["source_file"],
                "people": [
                    {
                        "track_id": p.get("track_id"),
                        "openpose_person_id": p["openpose_person_id"],
                        "mean_confidence": p["mean_confidence"],
                        "valid_keypoints": p["valid_keypoints"],
                        # [[x, y, c], ...] x 25, preserving BODY_25 order
                        "keypoints": [[round(x, 3), round(y, 3), round(c, 5)]
                                      for x, y, c in p["keypoints"]],
                        # Gap-filled skeleton (not seen by the pose model)
                        "interpolated": bool(p.get("interpolated")),
                    }
                    for p in f["people"]
                ],
            }
            for f in parsed["frames"]
        ],
    }
    dest.write_text(json.dumps(payload), encoding="utf-8")
    log.info(f"skeleton written: {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)",
             stage="skeleton")


def summarize(parsed: dict, sample_frames: int = 2) -> dict:
    """Compact structure for the UI - never the whole JSON."""
    frames = parsed["frames"]
    sample = []
    for f in frames[:sample_frames]:
        sample.append(
            {
                "frame": f["frame"],
                "people_count": len(f["people"]),
                "people": [
                    {
                        "track_id": p.get("track_id"),
                        "mean_confidence": p["mean_confidence"],
                        "valid_keypoints": p["valid_keypoints"],
                        "first_3_keypoints": [
                            {
                                "name": BODY_25_NAMES[i],
                                "x": round(p["keypoints"][i][0], 2),
                                "y": round(p["keypoints"][i][1], 2),
                                "confidence": round(p["keypoints"][i][2], 4),
                            }
                            for i in range(3)
                        ],
                    }
                    for p in f["people"][:2]
                ],
            }
        )
    return {
        "keypoint_format": "BODY_25",
        "num_keypoints": 25,
        "frame_count": parsed["frame_count"],
        "total_person_detections": parsed["total_person_detections"],
        "avg_people_per_frame": parsed["avg_people_per_frame"],
        "max_people_in_a_frame": parsed["max_people_in_a_frame"],
        "track_count": parsed.get("track_count", 0),
        "tracks": parsed.get("tracks", [])[:5],
        "tracker": parsed.get("tracker", "nearest-centroid"),
        "boxes_available": any(p.get("bbox") for f in parsed["frames"] for p in f["people"]),
        "json_versions": parsed["json_versions"],
        "per_person_array_shape": "25 x 3 (x, y, confidence) flat 75 in raw OpenPose JSON",
        "sample_frames": sample,
    }
