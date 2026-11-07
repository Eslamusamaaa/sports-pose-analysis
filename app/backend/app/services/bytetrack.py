"""ByteTrack multi-object tracker, adapted for OpenPose skeletons.

WHY REPLACE THE CENTROID TRACKER
--------------------------------
The previous tracker matched people by nearest centroid with a fixed distance
gate. It has no motion model and no notion of track confidence, so on real
kumite footage it failed in two measurable ways:

  * It produced 21 tracks for 2 fighters on a 4865-frame clip - identities were
    repeatedly lost and re-created, which shreds the temporal continuity a GCN
    depends on.
  * During a clinch, two centroids converge and the greedy nearest-match swaps
    the two athletes' identities. Every frame after the swap contributes
    mislabelled motion.

ByteTrack (Zhang et al., ECCV 2022) fixes both:

  1. A Kalman filter predicts where each track should be, so matching compares
     detections against a PREDICTION rather than the last seen position. Fast
     motion - exactly what karate is - stops breaking the association.
  2. Two-stage association. High-confidence detections are matched first; the
     leftover LOW-confidence detections are then matched against still-unmatched
     tracks. This is the core ByteTrack insight: a blurred or partly occluded
     athlete produces a low-confidence detection that a single-threshold tracker
     throws away, creating exactly the identity break we were seeing.

ADAPTATION TO POSE INPUT
------------------------
ByteTrack was designed for a detector emitting boxes. OpenPose emits keypoints,
so each person's box is derived from their valid joints and the detection score
is the mean joint confidence. Everything downstream is unchanged.

The tracker deliberately does NOT use appearance features (no re-identification
network). It is motion-only, which is fast and dependency-light, but means an
athlete who leaves frame for a long time returns as a new identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------- geometry ----
def keypoints_to_box(kps: list[tuple[float, float, float]],
                     min_conf: float = 0.1,
                     pad: float = 0.12) -> tuple[np.ndarray, float] | None:
    """Tight box around the confident joints, plus the mean joint confidence.

    Padding matters: a bare keypoint hull is much smaller than the person, and
    IoU between two such hulls drops off far too fast for reliable matching.
    """
    pts = [(x, y, c) for x, y, c in kps if c > min_conf and (x or y)]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    cs = np.array([p[2] for p in pts])
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= w * pad
    x2 += w * pad
    y1 -= h * pad
    y2 += h * pad
    return np.array([x1, y1, x2, y2], dtype=np.float64), float(cs.mean())


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


# ----------------------------------------------------------- Kalman filter ----
class BoxKalman:
    """Constant-velocity filter on (cx, cy, aspect, height).

    Tracking the centre/aspect/height parametrisation rather than raw corners
    keeps the box stable when a limb extends: a punch stretches the keypoint
    hull sideways, and filtering corners directly would make the box jitter and
    depress IoU exactly during the frames that matter most.
    """

    def __init__(self, box: np.ndarray) -> None:
        cx, cy, a, h = self._to_xyah(box)
        self.mean = np.array([cx, cy, a, h, 0.0, 0.0, 0.0, 0.0])
        self.cov = np.diag([10.0, 10.0, 1e-2, 10.0, 100.0, 100.0, 1e-4, 100.0])
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8)
        self._q = np.diag([1.0, 1.0, 1e-4, 1.0, 10.0, 10.0, 1e-6, 10.0])
        self._r = np.diag([1.0, 1.0, 1e-2, 1.0])

    @staticmethod
    def _to_xyah(b: np.ndarray) -> np.ndarray:
        w, h = max(b[2] - b[0], 1e-3), max(b[3] - b[1], 1e-3)
        return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])

    @staticmethod
    def _to_xyxy(s: np.ndarray) -> np.ndarray:
        cx, cy, a, h = s[:4]
        w = max(a * h, 1e-3)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + self._q
        return self._to_xyxy(self.mean)

    def update(self, box: np.ndarray) -> None:
        z = self._to_xyah(box)
        y = z - self._H @ self.mean
        s = self._H @ self.cov @ self._H.T + self._r
        k = self.cov @ self._H.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ y
        self.cov = (np.eye(8) - k @ self._H) @ self.cov

    @property
    def box(self) -> np.ndarray:
        return self._to_xyxy(self.mean)


# ------------------------------------------------------------------ track ----
@dataclass
class Track:
    track_id: int
    kf: BoxKalman
    score: float
    state: str = "tentative"          # tentative -> confirmed -> lost -> removed
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    start_frame: int = 0
    # Running stats, used to rank athletes against officials/spectators.
    motion_sum: float = 0.0
    conf_sum: float = 0.0
    frames_seen: int = 0
    _last_centre: tuple[float, float] | None = field(default=None, repr=False)

    def mark_seen(self, box: np.ndarray, score: float) -> None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if self._last_centre is not None:
            self.motion_sum += float(np.hypot(cx - self._last_centre[0],
                                              cy - self._last_centre[1]))
        self._last_centre = (cx, cy)
        self.conf_sum += score
        self.frames_seen += 1

    @property
    def motion_per_frame(self) -> float:
        return self.motion_sum / max(self.frames_seen, 1)

    @property
    def mean_confidence(self) -> float:
        return self.conf_sum / max(self.frames_seen, 1)


class ByteTracker:
    """Motion-only ByteTrack over per-frame skeleton detections."""

    def __init__(self,
                 high_thresh: float = 0.45,
                 low_thresh: float = 0.15,
                 match_thresh: float = 0.75,
                 second_match_thresh: float = 0.55,
                 max_age: int = 30,
                 tentative_age: int = 5,
                 min_hits: int = 2,
                 iou_weight: float = 0.5,
                 dist_scale: float = 1.5) -> None:
        # match_thresh is a COST ceiling (1 - IoU), so 0.75 accepts IoU >= 0.25.
        # Karate needs this to be loose: a fighter's keypoint hull changes shape
        # dramatically between guard and full extension, and a strict IoU gate
        # breaks the track at exactly the moment a technique is thrown.
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_age = max_age
        # Grace period for a not-yet-confirmed track. Must be > 0: detections
        # flicker on crowded footage and a zero-tolerance policy prevents any
        # track from ever reaching min_hits.
        self.tentative_age = tentative_age
        self.min_hits = min_hits
        # 0.5 = equal weight to IoU and size-normalised centre distance.
        self.iou_weight = iou_weight
        # A detection one box-diagonal away costs dist_scale^-1 of the range.
        self.dist_scale = dist_scale
        self._next_id = 0
        self.tracks: list[Track] = []
        self.frame = 0

    # Pairs worse than the threshold are made effectively infinite BEFORE the
    # assignment runs. Filtering afterwards (the obvious implementation, and the
    # one used in the first version of this file) lets the optimiser "spend" a
    # detection on a pair that is then rejected, leaving a perfectly good track
    # unmatched - which spawns a duplicate identity. That alone accounted for
    # ~0.34 new tracks per frame, i.e. ~1650 spurious identities per clip.
    _BIG = 1e6

    @staticmethod
    def _match(cost: np.ndarray, thresh: float):
        """Hungarian assignment with infeasible pairs excluded up front."""
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        gated = np.where(cost <= thresh, cost, ByteTracker._BIG)
        rows, cols = linear_sum_assignment(gated)
        matches, taken_r, taken_c = [], set(), set()
        for r, c in zip(rows, cols):
            if gated[r, c] < ByteTracker._BIG:
                matches.append((int(r), int(c)))
                taken_r.add(int(r))
                taken_c.add(int(c))
        ur = [i for i in range(cost.shape[0]) if i not in taken_r]
        uc = [j for j in range(cost.shape[1]) if j not in taken_c]
        return matches, ur, uc

    def update(self, detections: list[tuple[np.ndarray, float]]) -> dict[int, int]:
        """Advance one frame. Returns {detection_index: track_id}."""
        self.frame += 1

        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        high = [(i, b, s) for i, (b, s) in enumerate(detections) if s >= self.high_thresh]
        low = [(i, b, s) for i, (b, s) in enumerate(detections)
               if self.low_thresh <= s < self.high_thresh]

        active = [t for t in self.tracks if t.state in ("tentative", "confirmed")]
        lost = [t for t in self.tracks if t.state == "lost"]
        assignment: dict[int, int] = {}

        def absorb(track: Track, det_i: int, box: np.ndarray, score: float) -> None:
            track.kf.update(box)
            track.mark_seen(box, score)
            track.score = score
            track.hits += 1
            track.time_since_update = 0
            track.state = "confirmed" if track.hits >= self.min_hits else "tentative"
            assignment[det_i] = track.track_id

        def associate(tracks: list[Track], dets: list, thresh: float):
            if not tracks or not dets:
                return list(range(len(tracks))), list(range(len(dets)))
            tb = np.array([t.kf.box for t in tracks])
            db = np.array([b for _, b, _ in dets])
            # Pure IoU association fails on this footage. MEASURED: a wide 1080p
            # tournament shot contains ~8 people, most of them small and distant
            # (officials, spectators). For a small box, a few pixels of motion
            # drops IoU below any usable gate, so the track dies and a duplicate
            # is created - 2030 identities for a 4865-frame clip, against 21 from
            # the naive centroid tracker.
            #
            # The cost therefore blends IoU with centre distance normalised by
            # box size, which stays meaningful regardless of how small the
            # subject is. IoU still dominates when boxes genuinely overlap.
            cost = 1.0 - iou_matrix(tb, db)
            tc = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1)
            dc = np.stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2], 1)
            dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=-1)
            scale = np.maximum(
                np.hypot(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])[:, None], 1.0)
            dist_cost = np.clip(dist / (self.dist_scale * scale), 0.0, 1.0)
            cost = self.iou_weight * cost + (1.0 - self.iou_weight) * dist_cost
            m, ut, ud = self._match(cost, thresh)
            for ti, di in m:
                absorb(tracks[ti], *dets[di])
            return ut, ud

        # --- stage 1: high-confidence detections vs ACTIVE tracks only ---
        # Lost tracks are deliberately excluded here. Including them (as the
        # first version did) lets a stale track whose Kalman prediction has
        # drifted win a detection away from a healthy active track; the active
        # track then ages out and a duplicate identity is created.
        ut, ud = associate(active, high, self.match_thresh)
        remaining = [active[i] for i in ut]

        # --- stage 2: the ByteTrack step - LOW-confidence detections ---
        # A blurred or partly occluded athlete scores low. A single-threshold
        # tracker discards that detection and breaks the identity; here it is
        # used to keep the track alive.
        ut2, _ = associate(remaining, low, self.second_match_thresh)

        # --- stage 3: leftover HIGH detections vs LOST tracks (re-find) ---
        # An athlete who was occluded reappears here and recovers their original
        # id instead of being issued a new one.
        leftover_high = [high[i] for i in ud]
        # ud3 indexes leftover_high, which is what the new-track loop reads.
        _, still_unmatched = associate(lost, leftover_high, self.match_thresh)

        # --- unmatched tracks age out ---
        #
        # MEASURED BUG (fixed here): killing a tentative track on its first
        # missed frame destroyed the tracker on crowded footage. With ~8 people
        # per frame OpenPose detections flicker, so almost no track survived the
        # min_hits window; the tracker spawned ~2400 identities for a 4865-frame
        # clip and the longest surviving track covered only 1271 frames - worse
        # than the naive centroid tracker it was meant to replace.
        #
        # Tentative tracks now get their own (shorter) grace period instead.
        unmatched = [remaining[i] for i in ut2] + [t for t in lost
                     if t.time_since_update > 0 and t.state == "lost"]
        for t in unmatched:
            if t.time_since_update == 0:
                continue                      # was matched this frame
            limit = self.max_age if t.state != "tentative" else self.tentative_age
            if t.time_since_update > limit:
                t.state = "removed"
            elif t.state != "tentative":
                t.state = "lost"

        # --- unmatched HIGH-confidence detections start new tracks ---
        # Only high-confidence ones: spawning tracks from low-confidence noise is
        # how the old tracker ended up with 21 identities for 2 people.
        for di in still_unmatched:
            det_i, box, score = leftover_high[di]
            t = Track(track_id=self._next_id, kf=BoxKalman(box), score=score,
                      start_frame=self.frame)
            t.mark_seen(box, score)
            self._next_id += 1
            self.tracks.append(t)
            assignment[det_i] = t.track_id

        self.tracks = [t for t in self.tracks if t.state != "removed"]
        return assignment

    def ranked_tracks(self) -> list[dict]:
        """Tracks ordered by how much they look like a competing athlete.

        Presence alone selects stationary referees - measured on real footage,
        officials moved 5.6 px/frame while fighters moved 16-21. Motion is
        therefore a first-class term, not a tiebreaker.
        """
        out = []
        mx = max((t.motion_per_frame for t in self.tracks), default=0.0) or 1.0
        for t in self.tracks:
            out.append({
                "track_id": t.track_id,
                "frames_present": t.frames_seen,
                "mean_confidence": round(t.mean_confidence, 4),
                "motion_per_frame": round(t.motion_per_frame, 2),
                "score": round(t.mean_confidence * t.frames_seen
                               * (0.25 + 0.75 * t.motion_per_frame / mx), 4),
            })
        out.sort(key=lambda d: d["score"], reverse=True)
        return out


def track_sequence(frames: list[dict], log=None,
                   **kwargs) -> tuple[list[dict], list[dict]]:
    """Run ByteTrack across parsed OpenPose frames, assigning 'track_id' in place.

    `frames` are the dicts produced by skeleton_service.parse_openpose_dir.
    """
    tracker = ByteTracker(**kwargs)
    stats: dict[int, dict] = {}

    for frame in frames:
        dets: list[tuple[np.ndarray, float]] = []
        det_to_person: list[int] = []
        for pi, person in enumerate(frame["people"]):
            # The detector's own box when the engine has one (RTMDet / YOLO):
            # tighter and steadier than a hull around the joints.
            if person.get("bbox"):
                got = (np.asarray(person["bbox"], dtype=float),
                       float(person.get("bbox_score") or 0.5))
            else:
                got = keypoints_to_box(person["keypoints"])
            if got is None:
                person["track_id"] = None
                continue
            dets.append(got)
            det_to_person.append(pi)

        assignment = tracker.update(dets)

        for pi in range(len(frame["people"])):
            frame["people"][pi].setdefault("track_id", None)
        for di, tid in assignment.items():
            frame["people"][det_to_person[di]]["track_id"] = tid
            st = stats.setdefault(tid, {"frames": 0})
            st["frames"] += 1

    ranked = tracker.ranked_tracks()
    if log is not None:
        log.info(
            f"ByteTrack: {len(ranked)} tracks over {len(frames)} frames",
            stage="skeleton",
        )
        for t in ranked[:4]:
            log.info(
                f"  track {t['track_id']}: {t['frames_present']} frames, "
                f"conf {t['mean_confidence']}, motion {t['motion_per_frame']}px/f, "
                f"score {t['score']}",
                stage="skeleton",
            )
    return frames, ranked
