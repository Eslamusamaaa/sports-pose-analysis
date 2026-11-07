"""Runs the OpenPose demo binary as a subprocess.

Notes learned from the real binary (v1.7.0 CPU portable release):

* It MUST be launched with cwd = the OpenPose root. Running it from bin/ makes
  it fail to locate its resources.
* ``--model_folder`` must end with a path separator, and the value is passed as
  a separate argv entry (never string-concatenated into a shell line) so paths
  containing spaces - like this project's - survive intact.
* ``--frame_last N`` is inclusive, so it yields N+1 frames.
* Video output is MJPG-in-AVI and needs transcoding before a browser can play
  it (handled by video_service.to_browser_mp4).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from ..config import get_settings
from ..utils.logging_utils import JobLogger


class OpenPoseError(RuntimeError):
    pass


def preflight() -> list[str]:
    """Return a list of human-readable problems; empty means ready."""
    s = get_settings()
    problems: list[str] = []
    if not s.active_openpose_bin.exists():
        problems.append(f"OpenPose binary not found: {s.active_openpose_bin}")
    if not s.active_openpose_dir.exists():
        problems.append(f"OpenPose directory not found: {s.active_openpose_dir}")
    if not s.body25_model_file.exists():
        problems.append(
            f"BODY_25 weights not found: {s.body25_model_file} "
            "(models/getBaseModels.bat points at CMU's offline server; "
            "run scripts/download_models.ps1 instead)"
        )
    proto = s.openpose_model_dir / "pose" / "body_25" / "pose_deploy.prototxt"
    if not proto.exists():
        problems.append(f"BODY_25 prototxt not found: {proto}")
    return problems


def build_command(video: Path, json_dir: Path, out_video: Path,
                  max_frames: int | None = None) -> list[str]:
    s = get_settings()
    # model_folder must carry a trailing separator; forward slashes avoid the
    # backslash-before-quote escaping trap on Windows.
    model_folder = str(s.openpose_model_dir).replace("\\", "/").rstrip("/") + "/"
    cmd = [
        str(s.active_openpose_bin),
        "--video", str(video),
        "--model_folder", model_folder,
        "--model_pose", "BODY_25",
        "--net_resolution", s.openpose_net_resolution,
        "--write_json", str(json_dir),
        "--display", "0",
        "--render_pose", str(s.openpose_render_pose),
    ]
    if s.openpose_render_pose != 0:
        cmd += ["--write_video", str(out_video)]
    limit = s.effective_max_frames if max_frames is None else max_frames
    if limit and limit > 0:
        # inclusive -> subtract 1 so we get exactly `limit` frames
        cmd += ["--frame_last", str(max(limit - 1, 0))]
    return cmd


def run(video: Path, json_dir: Path, out_video: Path, log: JobLogger,
        max_frames: int | None = None) -> dict:
    s = get_settings()
    problems = preflight()
    if problems:
        raise OpenPoseError("; ".join(problems))

    json_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale JSON so frame counts reflect only this run.
    for stale in json_dir.glob("*_keypoints.json"):
        stale.unlink()
    if out_video.exists():
        out_video.unlink()

    cmd = build_command(video, json_dir, out_video, max_frames)
    log.info("OpenPose command: " + " ".join(f'"{c}"' if " " in c else c for c in cmd),
             stage="openpose")
    log.info(f"OpenPose cwd: {s.active_openpose_dir}", stage="openpose")
    log.info(f"OpenPose started on {s.openpose_device_label}", stage="openpose")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(s.active_openpose_dir),  # required: must run from OpenPose root
            capture_output=True,
            text=True,
            timeout=s.openpose_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenPoseError(
            f"OpenPose timed out after {s.openpose_timeout_s}s. "
            "Lower OPENPOSE_MAX_FRAMES or OPENPOSE_NET_RESOLUTION."
        ) from exc
    elapsed = round(time.perf_counter() - t0, 2)

    stdout_tail = (proc.stdout or "").strip()[-2000:]
    stderr_tail = (proc.stderr or "").strip()[-4000:]
    for line in stdout_tail.splitlines()[-12:]:
        log.debug(f"openpose: {line}", stage="openpose")

    produced = sorted(json_dir.glob("*_keypoints.json"))
    if proc.returncode != 0 and not produced:
        log.error(f"OpenPose exited {proc.returncode}", stage="openpose")
        raise OpenPoseError(
            f"OpenPose failed (exit {proc.returncode}).\n"
            f"--- stderr ---\n{stderr_tail or '(empty)'}\n"
            f"--- stdout ---\n{stdout_tail or '(empty)'}"
        )
    if not produced:
        raise OpenPoseError(
            "OpenPose produced no JSON files. "
            f"stderr: {stderr_tail or '(empty)'}"
        )

    log.info(
        f"OpenPose completed in {elapsed}s, {len(produced)} JSON frames written",
        stage="openpose",
    )
    return {
        "command": cmd,
        "cwd": str(s.active_openpose_dir),
        "exit_code": proc.returncode,
        "duration_s": elapsed,
        "json_frame_count": len(produced),
        "json_dir": str(json_dir),
        "video_written": out_video.exists(),
        "net_resolution": s.openpose_net_resolution,
        "model_pose": "BODY_25",
        "device": s.openpose_device_label,
        "stderr_tail": stderr_tail[-1200:],
        "seconds_per_frame": round(elapsed / max(len(produced), 1), 3),
    }


def openpose_version() -> str | None:
    s = get_settings()
    vf = s.active_openpose_dir / "Version.txt"
    if vf.exists():
        try:
            # utf-8-sig: the shipped Version.txt starts with a BOM.
            return vf.read_text(encoding="utf-8-sig").strip().splitlines()[0]
        except OSError:
            return None
    return None
