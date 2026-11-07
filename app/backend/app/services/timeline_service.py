"""Sliding-window action recognition over a whole clip.

WHY THIS EXISTS
---------------
A single call to ``inference_skeleton`` collapses the ENTIRE clip into one
label. Internally the config's ``UniformSampleFrames(clip_len=100)`` picks 100
frames spread evenly across however many frames you hand it - so for a
1800-frame (60 s) video it samples roughly one frame every 0.6 s.

That is not random, but it is fatal for fast sport: a karate punch lasts about
0.3 s (~9 frames), so it falls *between* samples and effectively disappears. The
model then sees "people drifting around over a minute" and answers accordingly -
on a real match it returned "walking towards each other" at 96.5% confidence.

This module instead splits the clip into overlapping windows of ``clip_len``
frames and classifies EVERY window, so no frame is skipped. The result is a
timeline of predictions rather than one averaged verdict, which is also the
shape you actually want for reviewing a match.

Cost note: windows are classified by the PERSISTENT worker, which loads the
recognizer once. Using the one-shot path would reload the model for every
window - for a 5000-frame clip at stride 50 that is ~100 reloads.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger

WORKER = BACKEND_DIR / "mmaction_worker.py"
MSG_PREFIX = "@@MMA@@"


class TimelineError(RuntimeError):
    pass


class _WorkerSession:
    """Owns one persistent recognizer process."""

    def __init__(self, log: JobLogger, num_clips: int = 1, model: dict | None = None) -> None:
        s = get_settings()
        cfg = {
            "config": str(s.mmaction_config),
            "checkpoint": str(s.checkpoint_path),
            "label_map": str(s.label_map_path),
            "device": s.device,
            "num_clips": num_clips,
        }
        # model: optional {"config", "checkpoint", "label_map"} override, e.g.
        # the swimming checkpoint instead of the default NTU60 one.
        cfg.update({k: str(v) for k, v in (model or {}).items()})
        self.log = log
        self.proc = subprocess.Popen(
            [str(s.active_mmaction_python), str(WORKER), json.dumps(cfg)],
            cwd=str(s.mmaction2_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read(timeout=300)
        if not ready or not ready.get("ready"):
            err = ""
            try:
                err = (self.proc.stderr.read() or "")[-1500:]
            except OSError:
                pass
            raise TimelineError(f"MMAction2 worker failed to start. {err}")
        self.ready = ready
        log.info(
            f"recognizer loaded once on {ready.get('device_used')} "
            f"(num_clips={ready.get('num_clips')}, {ready.get('num_classes')} classes)",
            stage="mmaction",
        )

    def _read(self, timeout: float = 120.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                return None
            if line.startswith(MSG_PREFIX):
                return json.loads(line[len(MSG_PREFIX):])
        return None

    def classify(self, pose_results: list[dict], img_shape: tuple[int, int],
                 req_id: int, topk: int = 5) -> dict | None:
        req = {
            "id": req_id,
            "pose_results": pose_results,
            "img_shape": [int(img_shape[0]), int(img_shape[1])],
            "topk": topk,
            "return_scores": True,
        }
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise TimelineError(f"worker pipe closed: {exc}") from exc
        return self._read(timeout=300)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
            except OSError:
                pass


def analyse_timeline(
    pose_results: list[dict],
    img_shape: tuple[int, int],
    log: JobLogger,
    fps: float = 30.0,
    window: int | None = None,
    stride: int | None = None,
    topk: int = 5,
    model: dict | None = None,
) -> dict:
    """Classify every window of the clip and return a timeline + aggregate.

    window: frames per window (defaults to the config's clip_len of 100)
    stride: frames advanced between windows; < window gives overlap so an action
            straddling a boundary is still seen whole by some window.
    model:  optional checkpoint override passed through to the worker.
    """
    s = get_settings()
    window = window or s.timeline_window
    stride = stride or s.timeline_stride
    total = len(pose_results)
    if total == 0:
        raise TimelineError("no frames to analyse")

    # Window starts covering EVERY frame. The final window is pulled back so the
    # tail of the clip is analysed instead of silently dropped.
    starts: list[int] = list(range(0, max(total - window, 0) + 1, stride))
    if not starts:
        starts = [0]
    last_start = max(total - window, 0)
    if starts[-1] != last_start:
        starts.append(last_start)

    log.info(
        f"timeline: {total} frames -> {len(starts)} windows "
        f"(window={window}, stride={stride}, overlap={window - stride} frames); "
        "every frame is covered",
        stage="mmaction",
    )

    session = _WorkerSession(log, num_clips=s.realtime_num_clips, model=model)
    windows: list[dict] = []
    score_sum: dict[int, float] = defaultdict(float)
    label_of: dict[int, str] = {}
    t0 = time.perf_counter()

    try:
        for i, start in enumerate(starts):
            chunk = pose_results[start:start + window]
            if not chunk:
                continue
            resp = session.classify(chunk, img_shape, req_id=i, topk=topk)
            if not resp:
                raise TimelineError("worker stopped responding mid-timeline")
            if not resp.get("ok"):
                raise TimelineError(resp.get("error", "unknown worker error"))

            end = start + len(chunk) - 1
            top1 = resp.get("top1") or {}
            windows.append(
                {
                    "index": i,
                    "start_frame": start,
                    "end_frame": end,
                    "start_s": round(start / fps, 2) if fps else None,
                    "end_s": round((end + 1) / fps, 2) if fps else None,
                    "top1": top1,
                    "topk": resp.get("topk", []),
                    "inference_s": resp.get("inference_s"),
                }
            )
            for t in resp.get("topk", []):
                label_of[t["class_index"]] = t["label"]
            for idx, v in enumerate(resp.get("scores") or []):
                score_sum[idx] += v

            if (i + 1) % 20 == 0 or i + 1 == len(starts):
                log.info(f"  classified {i + 1}/{len(starts)} windows", stage="mmaction")
    finally:
        session.close()

    elapsed = round(time.perf_counter() - t0, 2)

    # Aggregate 1: mean score per class across all windows.
    n_win = max(len(windows), 1)
    mean_scores = sorted(
        ((idx, total_score / n_win) for idx, total_score in score_sum.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )[:topk]
    overall = [
        {
            "rank": r + 1,
            "class_index": idx,
            "label": label_of.get(idx, f"class_{idx}"),
            "score": round(sc, 6),
        }
        for r, (idx, sc) in enumerate(mean_scores)
    ]

    # Aggregate 2: how often each label won a window. More robust than the mean
    # when a clip contains several different actions.
    votes: dict[str, int] = defaultdict(int)
    for w in windows:
        lbl = (w["top1"] or {}).get("label")
        if lbl:
            votes[lbl] += 1
    ranked_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:topk]

    log.info(
        f"timeline completed: {len(windows)} windows in {elapsed}s "
        f"({elapsed / n_win:.3f}s per window); "
        f"overall top1='{overall[0]['label'] if overall else None}'",
        stage="mmaction",
    )

    return {
        "windows": windows,
        "window_frames": window,
        "stride_frames": stride,
        "window_seconds": round(window / fps, 2) if fps else None,
        "num_windows": len(windows),
        "frames_covered": total,
        "duration_s": elapsed,
        "overall_topk": overall,
        "top1": overall[0] if overall else None,
        "label_votes": [
            {"label": lbl, "windows_won": n, "share": round(n / n_win, 3)}
            for lbl, n in ranked_votes
        ],
    }
