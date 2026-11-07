"""Video validation, probing and transcoding (ffmpeg/ffprobe)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import get_settings
from ..utils.logging_utils import JobLogger

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


class VideoError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(video: Path) -> dict:
    """Return real container/stream metadata. Never guesses."""
    s = get_settings()
    cmd = [
        s.ffprobe_bin, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video),
    ]
    proc = _run(cmd, timeout=120)
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed: {proc.stderr.strip()[:500]}")
    data = json.loads(proc.stdout)
    vstream = next((st for st in data.get("streams", []) if st.get("codec_type") == "video"), None)
    if vstream is None:
        raise VideoError("No video stream found in the uploaded file.")

    # avg_frame_rate arrives as "30000/1001"; evaluate it rather than assuming.
    fps = 0.0
    raw_fps = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1"
    try:
        num, _, den = raw_fps.partition("/")
        fps = round(float(num) / float(den), 3) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    nb_frames = vstream.get("nb_frames")
    try:
        nb_frames = int(nb_frames) if nb_frames not in (None, "N/A") else None
    except ValueError:
        nb_frames = None

    duration = data.get("format", {}).get("duration")
    try:
        duration = round(float(duration), 3) if duration else None
    except ValueError:
        duration = None

    if nb_frames is None and duration and fps:
        nb_frames = int(round(duration * fps))

    # Phones store portrait video sideways plus a rotation flag that players
    # honour but OpenPose (OpenCV) ignores - it then sees people lying on their
    # side and misses most of them. Newer files use side data, older a tag.
    rotation = 0
    for sd in vstream.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if not rotation:
        try:
            rotation = int(float((vstream.get("tags") or {}).get("rotate", 0)))
        except ValueError:
            rotation = 0

    return {
        "rotation": rotation % 360,
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "fps": fps,
        "duration_s": duration,
        "frame_count": nb_frames,
        "codec": vstream.get("codec_name"),
        "pix_fmt": vstream.get("pix_fmt"),
        "size_bytes": video.stat().st_size,
        "format": data.get("format", {}).get("format_name"),
    }


def validate(video: Path, log: JobLogger) -> dict:
    if not video.exists() or video.stat().st_size == 0:
        raise VideoError("Uploaded file is empty or missing.")
    meta = probe(video)
    if not meta["width"] or not meta["height"]:
        raise VideoError("Could not determine video resolution.")
    log.info(
        f"video ok: {meta['width']}x{meta['height']} @ {meta['fps']}fps, "
        f"{meta['frame_count']} frames, codec={meta['codec']}, "
        f"{meta['size_bytes'] / 1e6:.1f} MB",
        stage="validate",
    )
    return meta


def make_upright(src: Path, dst: Path, rotation: int, log: JobLogger) -> bool:
    """Bake the rotation flag into the pixels (ffmpeg autorotates on decode)."""
    s = get_settings()
    cmd = [
        s.ffmpeg_bin, "-y", "-v", "error", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", "-metadata:s:v:0", "rotate=0", str(dst),
    ]
    proc = _run(cmd, timeout=1800)
    ok = proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    if ok:
        log.info(f"video had a {rotation} deg rotation flag; wrote upright copy "
                 f"{dst.name} so OpenPose sees people the right way up", stage="validate")
    else:
        log.warning(f"could not make upright copy: {proc.stderr.strip()[:300]}", stage="validate")
    return ok


def trim_frames(src: Path, dst: Path, max_frames: int, log: JobLogger) -> bool:
    """Cap the clip to max_frames so a CPU OpenPose run stays interactive.

    Returns True if a trimmed copy was written.
    """
    if max_frames <= 0:
        return False
    s = get_settings()
    cmd = [
        s.ffmpeg_bin, "-y", "-i", str(src),
        "-frames:v", str(max_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    log.info(f"trimming to first {max_frames} frames: {' '.join(cmd)}", stage="validate")
    proc = _run(cmd, timeout=900)
    if proc.returncode != 0 or not dst.exists():
        log.warning(f"trim failed, using full video: {proc.stderr.strip()[-400:]}", stage="validate")
        return False
    return True


def to_browser_mp4(src: Path, dst: Path, log: JobLogger) -> bool:
    """Transcode OpenPose's AVI output to H.264 mp4.

    OpenPose writes MJPG-in-AVI, which no browser <video> element can play, so
    the overlay video is unusable in the UI without this step.
    """
    if not src.exists():
        log.warning(f"no source video to transcode: {src}", stage="video")
        return False
    s = get_settings()
    cmd = [
        s.ffmpeg_bin, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(dst),
    ]
    log.info(f"transcoding overlay to mp4: {' '.join(cmd)}", stage="video")
    proc = _run(cmd, timeout=1800)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        log.error(f"mp4 transcode failed: {proc.stderr.strip()[-600:]}", stage="video")
        return False
    log.info(f"overlay mp4 written: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)",
             stage="video")
    return True
