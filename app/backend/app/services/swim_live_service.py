"""Live stroke recognition for the SWIMMING page.

Separate from the karate live session on purpose: it uses the swimming
checkpoint, one person, and its own timing. What differs and why:

* Time-based window. The swim model was trained on 60 frames at 30 fps = 2 s.
  The webcam path runs OpenPose at ~15-20 fps, so "the last 60 frames" would be
  3-4 s and every stroke would look slow. Frames are timestamped and the last
  2 s are resampled to exactly 60 frames before each prediction.
* Standing mode (default on). A person acting out a stroke in front of a
  webcam is upright; every SwimXYZ swimmer is horizontal. Rotating the
  skeleton 90 deg (head -> the side, like a swimmer seen from the pool side)
  puts the pose into the orientation the model knows. It is a stand-in for
  testing, not a substitute for real pool footage.
* No person / low confidence are reported as states, never as a stroke.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

from ..config import get_settings
from ..utils.logging_utils import JobLogger
from .realtime_service import MIN_PRESENT_SHARE, RealtimeSession, person_present

MODEL_FPS = 30.0

# COCO-17 indices used for the "is this swimming at all?" gate.
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
ARM_JOINTS = (L_ELB, R_ELB, L_WRI, R_WRI)
# The model has no "not swimming" class - it always picks one of four strokes
# (a still person in front of the webcam came out "backstroke 61%"). So two
# checks run BEFORE the model:
#   arms visible: an elbow or wrist is found in at least this share of frames
MIN_ARM_VISIBLE = 0.4
#   arms moving:  a wrist/elbow travels at least this many shoulder-widths in
#                 the window. Any stroke moves the hands well over one shoulder
#                 width per 2 s; standing still or small fidgets stay far below.
MIN_ARM_TRAVEL = 0.8


def arm_activity(window: list[dict]) -> tuple[float, float]:
    """(share of frames with an arm joint, max arm travel in shoulder widths)."""
    seen = 0
    tracks: dict[int, list[tuple[float, float]]] = {j: [] for j in ARM_JOINTS}
    widths: list[float] = []
    for e in window:
        kp, sc = e["keypoints"][0], e["keypoint_scores"][0]
        if any(sc[j] > 0.05 for j in ARM_JOINTS):
            seen += 1
        for j in ARM_JOINTS:
            if sc[j] > 0.05:
                tracks[j].append((kp[j][0], kp[j][1]))
        if sc[L_SHO] > 0.05 and sc[R_SHO] > 0.05:
            widths.append(((kp[L_SHO][0] - kp[R_SHO][0]) ** 2
                           + (kp[L_SHO][1] - kp[R_SHO][1]) ** 2) ** 0.5)
    scale = sorted(widths)[len(widths) // 2] if widths else 0.0
    travel = 0.0
    for pts in tracks.values():
        if len(pts) >= 5 and scale > 1:
            path = sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
                       for i in range(1, len(pts)))
            travel = max(travel, path / scale)
    return seen / max(len(window), 1), travel


def _rotate_entry(entry: dict, w: int) -> dict:
    """Rotate 90 deg: (x, y) -> (y, w-1-x); frame (h, w) becomes (w, h)."""
    kps = []
    for person, scores in zip(entry["keypoints"], entry["keypoint_scores"]):
        kps.append([
            [y, (w - 1) - x] if c > 0 else [0.0, 0.0]
            for (x, y), c in zip(person, scores)
        ])
    return {"keypoints": kps, "keypoint_scores": entry["keypoint_scores"]}


class SwimRealtimeSession(RealtimeSession):
    sport = "swimming"

    def __init__(self, session_dir: Path, log: JobLogger, standing: bool = True,
                 engine: str = "openpose") -> None:
        super().__init__(session_dir, log, engine=engine)
        s = get_settings()
        self.max_persons = 1
        self.standing = standing
        self.window_s = s.swim_window / MODEL_FPS  # 2.0 s
        self.min_conf = s.swim_min_confidence
        self.frames: deque[tuple[float, dict]] = deque(maxlen=240)
        self.state.update(
            {
                "window": s.swim_window,
                "window_seconds": self.window_s,
                "standing_mode": standing,
                "stroke": None,
                "confidence": None,
            }
        )

    def _model_cfg(self) -> dict:
        s = get_settings()
        return {
            "config": str(s.swim_config),
            "checkpoint": str(s.swim_checkpoint),
            "label_map": str(s.swim_label_map),
        }

    def start(self) -> None:
        if not get_settings().swim_model_ready:
            raise RuntimeError("Swimming model is not installed (models/swim/).")
        super().start()

    # ------------------------------------------------------------------ frames
    def _on_frame(self, entry: dict, frame_no: int) -> None:
        now = time.time()
        self.frames.append((now, entry))
        while self.frames and now - self.frames[0][0] > self.window_s + 0.5:
            self.frames.popleft()
        span = now - self.frames[0][0] if self.frames else 0.0
        with self._lock:
            self.state["buffer_fill"] = min(
                int(round(span / self.window_s * self.state["window"])), self.state["window"]
            )
        if span >= 0.9 * self.window_s and frame_no % self.infer_every == 0:
            self._infer_window(now)

    def _resample(self, now: float) -> list[dict]:
        """Last window_s seconds -> exactly `window` frames at 30 fps (nearest)."""
        n = self.state["window"]
        pts = [(t, e) for t, e in self.frames if now - t <= self.window_s + 1e-6]
        out, j = [], 0
        for k in range(n):
            target = now - self.window_s + (k + 0.5) * self.window_s / n
            while j + 1 < len(pts) and abs(pts[j + 1][0] - target) <= abs(pts[j][0] - target):
                j += 1
            out.append(pts[j][1])
        return out

    def _infer_window(self, now: float) -> None:
        window = self._resample(now)
        present = sum(person_present(e) for e in window) / max(len(window), 1)
        if present < MIN_PRESENT_SHARE:
            with self._lock:
                self.state.update(status="no_person", prediction=None, stroke=None, confidence=None)
            return

        arms_seen, travel = arm_activity(window)
        with self._lock:
            self.state["arm_visible_share"] = round(arms_seen, 2)
            self.state["arm_travel"] = round(travel, 2)
        if arms_seen < MIN_ARM_VISIBLE:
            with self._lock:
                self.state.update(status="arms_not_visible", prediction=None, stroke=None, confidence=None)
            return
        if travel < MIN_ARM_TRAVEL:
            with self._lock:
                self.state.update(status="not_swimming", prediction=None, stroke=None, confidence=None)
            return

        h, w = self.frame_hw or get_settings().realtime_frame_hw
        if self.standing:
            window = [_rotate_entry(e, w) for e in window]
            h, w = w, h

        self._req_id += 1
        req = {"id": self._req_id, "pose_results": window, "img_shape": [h, w], "topk": 4}
        try:
            self._mm.stdin.write(json.dumps(req) + "\n")
            self._mm.stdin.flush()
            resp = self._read_msg(timeout=60)
        except (BrokenPipeError, OSError, AttributeError) as exc:
            with self._lock:
                self.state["error"] = f"swim worker pipe closed: {exc}"
            return
        if not resp:
            return
        with self._lock:
            if not resp.get("ok"):
                self.state["error"] = resp.get("error")
                return
            top1 = resp.get("top1") or {}
            conf = float(top1.get("score") or 0.0)
            self.state["prediction"] = {"top1": top1, "topk": resp.get("topk"), "num_clips": 1}
            self.state["inference_s"] = resp.get("inference_s")
            self.state["confidence"] = round(conf, 4)
            if conf >= self.min_conf:
                self.state.update(status="ok", stroke=top1.get("label"))
            else:
                self.state.update(status="uncertain", stroke=None)
