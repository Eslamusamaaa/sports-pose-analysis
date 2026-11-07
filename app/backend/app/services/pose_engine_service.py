"""Alternative pose engines (RTMPose+RTMDet, YOLO11-pose) for BOTH sports.

Runs backend/pose_extract.py in the MMAction2 venv and returns the same dict
shape as openpose_service.run, so the pipeline and UI treat every engine the
same way. The engine writes OpenPose-format JSON, so tracking, BODY_25 ->
COCO-17 and MMAction2 are untouched.

Measured on our own clips (same frames, conf 0.3):
  kumite 1080p   RTMPose 100% / 0.19 s/frame   YOLO11 100% / 0.03 s/frame
  swim (front)   RTMPose  69%                  YOLO11  34%   (OpenPose 25%)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger

SCRIPT = BACKEND_DIR / "pose_extract.py"
MSG = "@@POSE@@"

# Typical GPU seconds per frame on this machine (RTX 3050 4 GB), measured on
# our own clips; only used for the "estimated time" line in the job log.
TYPICAL_S_PER_FRAME = {"openpose": 0.08, "rtmpose": 0.15, "yolo11": 0.06}

ENGINES = {
    "openpose": {"label": "OpenPose BODY_25", "keypoints": "BODY_25 (25)", "license": "CMU academic / non-commercial"},
    "rtmpose": {"label": "RTMPose-m + RTMDet-m", "keypoints": "Halpe26 (COCO-17 + feet)", "license": "Apache-2.0"},
    "yolo11": {"label": "YOLO11m-pose", "keypoints": "COCO-17 (no feet)", "license": "AGPL-3.0 (commercial needs a license)"},
}


class PoseEngineError(RuntimeError):
    pass


def run(engine: str, video: Path, json_dir: Path, out_video: Path, log: JobLogger,
        max_frames: int = 0) -> dict:
    s = get_settings()
    if not s.pose_engine_ready(engine):
        raise PoseEngineError(f"{engine} is not installed (libs/weights missing)")
    json_dir.mkdir(parents=True, exist_ok=True)
    for stale in json_dir.glob("*_keypoints.json"):
        stale.unlink()
    (json_dir.parent / "reid.npy").unlink(missing_ok=True)
    if out_video.exists():
        out_video.unlink()

    cfg = {
        "engine": engine,
        "video": str(video),
        "json_dir": str(json_dir),
        "out_video": str(out_video),
        "libs": [str(s.pose_libs_dir), str(s.yolo_libs_dir)],
        "models": s.pose_engine_models(engine),
        "device": s.device,
        "max_frames": max_frames or 0,
        "reid": s.reid_cfg,  # OSNet appearance embeddings (None if not installed)
    }
    label = ENGINES[engine]["label"]
    log.info(f"{label} started on {s.device}", stage=engine)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [str(s.active_mmaction_python), str(SCRIPT), json.dumps(cfg)],
        cwd=str(BACKEND_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    done, error = None, None
    for line in proc.stdout:  # progress streams into the job log as it happens
        if not line.startswith(MSG):
            continue
        msg = json.loads(line[len(MSG):])
        if msg.get("ready"):
            log.info(f"{label} loaded in {msg['load_s']}s; {msg['frames']} frames to process",
                     stage=engine)
        elif "progress" in msg:
            log.info(f"  {msg['progress']}/{msg['total']} frames", stage=engine)
        elif msg.get("done"):
            done = msg
        elif "error" in msg:
            error = msg["error"]
    try:
        proc.wait(timeout=s.openpose_timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise PoseEngineError(f"{label} timed out") from exc
    stderr_tail = (proc.stderr.read() or "")[-3000:] if proc.stderr else ""
    produced = sorted(json_dir.glob("*_keypoints.json"))
    if proc.returncode != 0 or not done or not produced:
        raise PoseEngineError(f"{label} failed: {error or 'no output'}\n{stderr_tail[-1500:]}")

    elapsed = round(time.perf_counter() - t0, 2)
    log.info(f"{label} completed: {len(produced)} frames in {done['duration_s']}s "
             f"({done['seconds_per_frame']} s/frame) + {done['load_s']}s model load", stage=engine)
    return {
        "engine": engine,
        "engine_label": label,
        "keypoint_source": ENGINES[engine]["keypoints"],
        "license": ENGINES[engine]["license"],
        "command": ["pose_extract.py", engine],
        "cwd": str(BACKEND_DIR),
        "exit_code": proc.returncode,
        "duration_s": elapsed,
        "json_frame_count": len(produced),
        "json_dir": str(json_dir),
        "video_written": out_video.exists(),
        "net_resolution": "n/a (top-down)" if engine == "rtmpose" else "640 (YOLO default)",
        "model_pose": "BODY_25 slots (mapped)",
        "device": s.device,
        "stderr_tail": stderr_tail[-800:],
        "seconds_per_frame": done["seconds_per_frame"],
        "model_load_s": done["load_s"],
    }


def reid_pass(video: Path, json_dir: Path, log: JobLogger) -> bool:
    """Add OSNet re-id looks to a JSON folder that has none (OpenPose writes
    only joints). Player identity then works the same for every pose engine.
    Non-fatal: returns False and the players are followed without looks."""
    s = get_settings()
    if not s.reid_cfg:
        log.warning("re-id model not installed; players followed by position only", stage="reid")
        return False
    cfg = {"mode": "reid", "video": str(video), "json_dir": str(json_dir),
           "device": s.device, "reid": s.reid_cfg}
    try:
        proc = subprocess.run([str(s.active_mmaction_python), str(SCRIPT), json.dumps(cfg)],
                              cwd=str(BACKEND_DIR), capture_output=True, text=True,
                              timeout=s.openpose_timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("re-id pass timed out; players followed by position only", stage="reid")
        return False
    done = next((json.loads(l[len(MSG):]) for l in proc.stdout.splitlines()
                 if l.startswith(MSG) and '"done"' in l), None)
    if proc.returncode != 0 or not done:
        log.warning(f"re-id pass failed: {(proc.stderr or '')[-400:]}", stage="reid")
        return False
    log.info(f"re-id looks added: {done['embeddings']} people over {done['frames']} frames "
             f"in {done['duration_s']}s", stage="reid")
    return True


def track_pass(json_dir: Path, width: int, height: int, log: JobLogger) -> bool:
    """Give tracker ids to engines that have none (RTMPose, OpenPose) with the
    same ByteTrack YOLO11 uses, fed with their boxes. Non-fatal."""
    s = get_settings()
    cfg = {"mode": "track", "json_dir": str(json_dir), "width": width, "height": height,
           "libs": [str(s.pose_libs_dir), str(s.yolo_libs_dir)]}
    try:
        proc = subprocess.run([str(s.active_mmaction_python), str(SCRIPT), json.dumps(cfg)],
                              cwd=str(BACKEND_DIR), capture_output=True, text=True,
                              timeout=s.openpose_timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("tracking pass timed out", stage="track")
        return False
    done = next((json.loads(l[len(MSG):]) for l in proc.stdout.splitlines()
                 if l.startswith(MSG) and '"done"' in l), None)
    if proc.returncode != 0 or not done:
        log.warning(f"tracking pass failed: {(proc.stderr or '')[-400:]}", stage="track")
        return False
    log.info(f"ByteTrack ids added: {done['tracks']} tracks over {done['frames']} frames", stage="track")
    return True


def estimate_s(engine: str, frames: int) -> int:
    return int(round(TYPICAL_S_PER_FRAME.get(engine, 0.08) * max(frames, 0)))


def device_label(engine: str) -> str:
    s = get_settings()
    if engine == "openpose":
        return s.openpose_device_label
    return f"{ENGINES[engine]['label']} on {s.device}"
