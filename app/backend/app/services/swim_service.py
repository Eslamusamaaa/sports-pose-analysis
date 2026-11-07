"""Swimming stroke recognition on top of the sliding-window timeline.

The swimming ST-GCN was fine-tuned on SwimXYZ (synthetic, CC-BY 4.0) with
ground-truth BODY_25 joints resampled to 30 fps and cut into contiguous
60-frame (2 s) windows. Inference must see the same thing, so:

  * a 50/60 fps upload is decimated to ~30 fps before windowing;
  * only ONE person (the top-ranked track) is sent - SwimXYZ has a single
    swimmer, and a second skeleton (coach, swimmer in the next lane) would be
    averaged into the prediction by GCNHead's person pooling.

The model has no "other" class (SwimXYZ has no dives, turns or standing), so a
window whose best score is below ``swim_min_confidence`` is reported as
``uncertain`` instead of being forced into one of the four strokes.

Coaching feedback (arm height, head position, kick depth/rate ...) is a later
phase; ``coaching`` is returned as an explicit placeholder so the UI and the
result schema already have a home for it.
"""
from __future__ import annotations

from collections import defaultdict

from ..config import get_settings
from ..utils.logging_utils import JobLogger
from . import timeline_service

TARGET_FPS = 30.0
UNCERTAIN = "uncertain"
NOT_VISIBLE = "not_visible"
# A frame "has the swimmer" when OpenPose found at least this many joints.
MIN_JOINTS_PER_FRAME = 5
# A window is classified only if the swimmer is present in this share of it.
# Measured on real pool clips: windows with less than this were mostly zero
# padding, and the model returned the same constant scores for all of them.
MIN_VISIBLE_SHARE = 0.5


SWIMMER_TRACK_ID = 9999
# A track counts as "the swimmer" if it moves at least this share of the
# fastest track's speed IN THE SAME VIDEO. Relative on purpose: pixel speeds
# depend on resolution, zoom and fps, so no absolute px/frame value is general.
MIN_SWIMMER_MOTION_SHARE = 0.4


def stitch_swimmer(parsed: dict, log: JobLogger) -> dict:
    """Join the swimmer's broken track pieces into one track (swimming only).

    A swimmer coming towards the camera grows and speeds up in the image, so the
    nearest-centroid tracker keeps losing them and opening new tracks. On a real
    clip it split one swimmer into 4 tracks (78 + 56 + 19 + 16 frames) and only
    the longest piece reached the model, although a person was detected in 207
    of 289 frames. Here every fast-moving track is treated as the swimmer;
    per frame the highest-ranked fast track present wins. Slow tracks (people on
    the deck) are left alone.
    """
    tracks = parsed.get("tracks") or []
    if not tracks:
        return parsed
    fastest = max(t.get("motion_per_frame", 0.0) for t in tracks)
    cut = MIN_SWIMMER_MOTION_SHARE * fastest
    pieces = [t for t in tracks if t.get("motion_per_frame", 0.0) >= cut]
    if len(pieces) < 2:
        return parsed
    rank = {t["track_id"]: i for i, t in enumerate(pieces)}
    covered = 0
    for frame in parsed["frames"]:
        cands = [p for p in frame["people"] if p.get("track_id") in rank]
        if not cands:
            continue
        best = min(cands, key=lambda p: rank[p["track_id"]])
        best["track_id"] = SWIMMER_TRACK_ID
        covered += 1
    merged = {
        "track_id": SWIMMER_TRACK_ID,
        "frames_present": covered,
        "mean_confidence": round(sum(t.get("mean_confidence", 0) for t in pieces) / len(pieces), 4),
        "motion_per_frame": round(max(t.get("motion_per_frame", 0) for t in pieces), 2),
        "score": max(t.get("score", 0) for t in pieces) + 1e6,  # always ranked first
        "stitched_from": [t["track_id"] for t in pieces],
    }
    parsed["tracks"] = [merged] + [t for t in tracks if t["track_id"] not in rank]
    log.info(
        f"swimming: stitched {len(pieces)} fast-moving track pieces "
        f"{merged['stitched_from']} into one swimmer track covering {covered} frames "
        f"(longest single piece had {pieces[0]['frames_present']})",
        stage="skeleton",
    )
    return parsed


def frame_visible(frame: dict) -> bool:
    scores = frame["keypoint_scores"][0] if frame.get("keypoint_scores") else []
    return sum(1 for c in scores if c > 0.05) >= MIN_JOINTS_PER_FRAME


def summarise(windows: list[dict]) -> tuple[str | None, list[dict]]:
    """Main stroke = highest summed score over VISIBLE windows.

    Counting window wins let two weak 'breaststroke' windows (0.64, 0.78) beat
    one clear 'butterfly' (0.97); summing the full score vectors weights each
    window by how sure the model actually was.
    """
    totals: dict[str, float] = defaultdict(float)
    wins: dict[str, int] = defaultdict(int)
    for w in windows:
        if w["stroke"] == NOT_VISIBLE:
            continue
        for k, v in w["scores"].items():
            totals[k] += v
        if w["stroke"] != UNCERTAIN:
            wins[w["stroke"]] += 1
    grand = sum(totals.values()) or 1.0
    distribution = sorted(
        (
            {"stroke": k, "share": round(v / grand, 3), "windows": wins.get(k, 0)}
            for k, v in totals.items()
        ),
        key=lambda d: d["share"],
        reverse=True,
    )
    return (distribution[0]["stroke"] if distribution else None), distribution


def _merge_segments(windows: list[dict]) -> list[dict]:
    """Collapse consecutive windows with the same stroke into segments."""
    segments: list[dict] = []
    for w in windows:
        if segments and segments[-1]["stroke"] == w["stroke"]:
            seg = segments[-1]
            seg["end_s"] = w["end_s"]
            seg["end_frame"] = w["end_frame"]
            seg["scores"].append(w["confidence"])
        else:
            segments.append(
                {
                    "stroke": w["stroke"],
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "start_frame": w["start_frame"],
                    "end_frame": w["end_frame"],
                    "scores": [w["confidence"]],
                }
            )
    for seg in segments:
        sc = seg.pop("scores")
        seg["mean_confidence"] = round(sum(sc) / len(sc), 4)
        seg["num_windows"] = len(sc)
    return segments


def analyse_swimming(
    pose_results: list[dict],
    img_shape: tuple[int, int],
    fps: float,
    log: JobLogger,
) -> dict:
    s = get_settings()
    if not s.swim_model_ready:
        raise RuntimeError(
            "Swimming model is not installed yet. Expected "
            f"{s.swim_config}, {s.swim_checkpoint} and {s.swim_label_map}."
        )

    fps = fps or TARGET_FPS
    step = max(1, round(fps / TARGET_FPS))
    poses = pose_results[::step]
    eff_fps = fps / step
    log.info(
        f"swimming: {len(pose_results)} frames @ {fps:.2f} fps -> every {step} frame(s) "
        f"= {len(poses)} frames @ {eff_fps:.2f} fps (model trained at 30 fps)",
        stage="mmaction",
    )

    tl = timeline_service.analyse_timeline(
        poses,
        img_shape,
        log,
        fps=eff_fps,
        window=s.swim_window,
        stride=s.swim_stride,
        topk=4,
        model={
            "config": s.swim_config,
            "checkpoint": s.swim_checkpoint,
            "label_map": s.swim_label_map,
        },
    )

    visible = [frame_visible(f) for f in poses]
    windows: list[dict] = []
    for w in tl["windows"]:
        top1 = w.get("top1") or {}
        conf = float(top1.get("score") or 0.0)
        span = visible[w["start_frame"]:w["end_frame"] + 1]
        vis_share = sum(span) / max(len(span), 1)
        if vis_share < MIN_VISIBLE_SHARE:
            stroke = NOT_VISIBLE
        elif conf >= s.swim_min_confidence:
            stroke = top1.get("label")
        else:
            stroke = UNCERTAIN
        windows.append(
            {
                "start_s": w["start_s"],
                "end_s": w["end_s"],
                # Back to ORIGINAL video frame numbers so the UI can seek.
                "start_frame": w["start_frame"] * step,
                "end_frame": w["end_frame"] * step,
                "stroke": stroke,
                "confidence": round(conf, 4),
                "visible_share": round(vis_share, 3),
                "scores": {t["label"]: t["score"] for t in w.get("topk", [])},
            }
        )

    counts: dict[str, int] = defaultdict(int)
    for w in windows:
        counts[w["stroke"]] += 1
    dominant, distribution = summarise(windows)
    n_vis = len(windows) - counts.get(NOT_VISIBLE, 0)

    log.info(
        f"swimming: swimmer visible in {sum(visible)}/{len(visible)} frames; "
        f"{n_vis}/{len(windows)} windows classified; main stroke={dominant}",
        stage="mmaction",
    )

    return {
        "dominant_stroke": dominant,
        "distribution": distribution,
        "uncertain_windows": counts.get(UNCERTAIN, 0),
        "not_visible_windows": counts.get(NOT_VISIBLE, 0),
        "visible_frames": sum(visible),
        "total_frames": len(visible),
        "num_windows": len(windows),
        "windows": windows,
        "segments": _merge_segments(windows),
        "min_confidence": s.swim_min_confidence,
        "window_seconds": round(s.swim_window / eff_fps, 2),
        "stride_seconds": round(s.swim_stride / eff_fps, 2),
        "input_fps": fps,
        "model_fps": round(eff_fps, 2),
        "frame_step": step,
        "duration_s": tl["duration_s"],
        "model": {
            "name": "ST-GCN",
            "dataset": "SwimXYZ (synthetic)",
            "classes": ["freestyle", "backstroke", "breaststroke", "butterfly"],
        },
        "domain_warning": (
            "Trained on SYNTHETIC swimmers (SwimXYZ). Accuracy on real pool footage "
            "has not been measured yet - check results against the video."
        ),
        "coaching": {
            "status": "planned",
            "message": "Technique feedback (arm recovery, head position, kick) is the next phase.",
            "tips": [],
        },
    }
