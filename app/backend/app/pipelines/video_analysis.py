"""End-to-end pipeline: video -> OpenPose -> BODY_25 -> COCO-17 -> MMAction2.

Stage isolation is the point here. OpenPose succeeding and MMAction2 failing is
a normal, reportable outcome ("partial"), not a broken job - that is what makes
each stage independently debuggable from the UI.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from ..config import get_settings
from ..jobs import Job, store
from ..services import (
    mmaction_service,
    openpose_service,
    player_follow,
    players_service,
    pose_engine_service,
    skeleton_converter,
    skeleton_service,
    swim_service,
    timeline_service,
    video_service,
)
from ..utils.logging_utils import JobLogger


def _write_result(job: Job) -> None:
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "filename": job.filename,
        "sport": job.sport,
        "pose_engine": job.pose_engine,
        "size_bytes": job.size_bytes,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "video_meta": job.video_meta,
        "openpose": job.openpose,
        "skeleton": job.skeleton,
        "conversion": job.conversion,
        "mmaction": job.mmaction,
        "stages": [
            {
                "name": s.name,
                "state": s.state,
                "duration_s": s.duration_s,
                "error": s.error,
            }
            for s in (job.stages[k] for k in job.stages)
        ],
    }
    job.result_json.parent.mkdir(parents=True, exist_ok=True)
    job.result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_video(job: Job) -> None:
    """Run the full pipeline for an already-uploaded job."""
    s = get_settings()
    log = JobLogger(job.job_id, job.dir)
    log.info(f"pipeline started for job {job.job_id} ({job.filename})")

    # ---------------- stage: validate ----------------
    store.start_stage(job, "validate")
    try:
        meta = video_service.validate(job.input_video, log)
        job.video_meta = meta
        store.finish_stage(job, "validate", "done")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        log.error(f"video validation failed: {msg}", stage="validate")
        store.finish_stage(job, "validate", "failed", msg)
        store.set_status(job, "failed", "validate", msg)
        _write_result(job)
        return

    analysis_input = job.input_video
    rotation = meta.get("rotation") or 0
    # Swimming only for now: the karate page is kept exactly as it was.
    if job.sport == "swimming" and rotation in (90, 180, 270):
        upright = job.dir / "input_upright.mp4"
        if video_service.make_upright(job.input_video, upright, rotation, log):
            analysis_input = upright
            job.video_meta["analysis_input"] = upright.name
            if rotation in (90, 270):
                # Keypoints come back in the rotated frame's pixel space, so the
                # model's img_shape must use the swapped size too.
                job.video_meta["width"], job.video_meta["height"] = meta["height"], meta["width"]

    # Cap frames so an OpenPose run returns within a browser-friendly time.
    limit = s.effective_max_frames
    frame_count = meta.get("frame_count") or 0

    # --frame_last is only passed when it is genuinely needed. OpenPose ABORTS
    # if --frame_last >= the clip length ("The desired last frame must be lower
    # than the length of the video"), so a cap larger than the video must never
    # reach the command line.
    openpose_frame_cap = 0  # 0 = omit the flag entirely

    if limit and frame_count and frame_count > limit:
        trimmed = job.dir / "input_trimmed.mp4"
        if video_service.trim_frames(analysis_input, trimmed, limit, log):
            # The trimmed file is already exactly `limit` frames long, so no flag.
            analysis_input = trimmed
            job.video_meta["analysis_frame_limit"] = limit
            job.video_meta["analysis_input"] = trimmed.name
        else:
            # Trim failed - fall back to the flag, clamped below the clip length.
            openpose_frame_cap = min(limit, frame_count)
            job.video_meta["analysis_frame_limit"] = openpose_frame_cap
        log.info(
            f"analysing first {limit} of {frame_count} frames "
            f"({pose_engine_service.device_label(job.pose_engine)}); est. {pose_engine_service.estimate_s(job.pose_engine, limit)}s. "
            "Set the frame cap to 0 for the full clip",
            stage="validate",
        )
    elif limit and not frame_count:
        # Unknown length: cap defensively, but we cannot clamp, so let OpenPose
        # read to the end instead of risking the abort above.
        log.info(
            f"frame count unknown; processing the whole clip with "
            f"{pose_engine_service.device_label(job.pose_engine)}",
            stage="validate",
        )
    else:
        log.info(
            f"clip is {frame_count} frames, within the {limit or 'unlimited'} cap; "
            f"processing all of it with {pose_engine_service.device_label(job.pose_engine)} "
            f"(est. {pose_engine_service.estimate_s(job.pose_engine, frame_count)}s)",
            stage="validate",
        )

    # ---------------- stage: openpose ----------------
    store.start_stage(job, "openpose")
    try:
        if job.pose_engine in ("rtmpose", "yolo11"):
            # Same OpenPose-format JSON + skeleton video, so every later stage
            # is identical whichever engine produced the keypoints.
            op = pose_engine_service.run(
                job.pose_engine,
                analysis_input,
                job.openpose_json_dir,
                job.openpose_video_avi,
                log,
                max_frames=openpose_frame_cap,
            )
            op["openpose_version"] = op["engine_label"]
        else:
            op = openpose_service.run(
                analysis_input,
                job.openpose_json_dir,
                job.openpose_video_avi,
                log,
                max_frames=openpose_frame_cap,
            )
            op["openpose_version"] = openpose_service.openpose_version()
            op["engine"], op["engine_label"] = "openpose", "OpenPose BODY_25"
            (job.openpose_json_dir.parent / "reid.npy").unlink(missing_ok=True)
        # Karate: Player 1/2 are followed by their LOOK as well as position, so
        # every engine needs re-id looks; OpenPose gives only joints, add them.
        if job.sport != "swimming" and not (job.openpose_json_dir.parent / "reid.npy").exists():
            pose_engine_service.reid_pass(analysis_input, job.openpose_json_dir, log)
        # ...and tracker ids: RTMPose / OpenPose give none, and our simple
        # nearest-centroid matcher cannot cope with a crowded hall.
        if job.sport != "swimming" and job.pose_engine != "yolo11":
            pose_engine_service.track_pass(job.openpose_json_dir, job.video_meta.get("width") or 0,
                                           job.video_meta.get("height") or 0, log)
        # Browsers cannot play OpenPose's MJPG/AVI; make an H.264 copy.
        if job.openpose_video_avi.exists():
            op["mp4_written"] = video_service.to_browser_mp4(
                job.openpose_video_avi, job.openpose_video_mp4, log
            )
        job.openpose = op
        store.finish_stage(job, "openpose", "done")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        engine_name = pose_engine_service.ENGINES.get(job.pose_engine, {}).get("label", job.pose_engine)
        log.error(f"{engine_name} failed: {msg}", stage=job.pose_engine)
        log.debug(traceback.format_exc(), stage=job.pose_engine)
        store.finish_stage(job, "openpose", "failed", msg)
        store.set_status(job, "failed", "openpose", msg)
        _write_result(job)
        return

    # ---------------- stage: skeleton ----------------
    store.start_stage(job, "skeleton")
    try:
        parsed = skeleton_service.parse_openpose_dir(job.openpose_json_dir, log)
        if s.tracker_backend == "bytetrack":
            from ..services import bytetrack
            _, ranked = bytetrack.track_sequence(parsed["frames"], log)
            parsed["tracks"] = ranked
            parsed["track_count"] = len(ranked)
        else:
            parsed = skeleton_service.track_people(
                parsed, log, rank_by="motion" if job.sport == "swimming" else "limbs"
            )
        if job.sport == "swimming":
            parsed = swim_service.stitch_swimmer(parsed, log)
        # Tag the players. Karate: by clothing on every frame (keeps the coach
        # out and survives tracker id swaps in clinches). Falls back to joining
        # track pieces by time/position when there are no clothing signatures
        # (OpenPose engine) - and swimming always uses that, since a swimmer's
        # "clothes" are mostly water and splash.
        identified = None
        if job.sport != "swimming":
            try:
                # Pick the two players once at the start, then follow those
                # same two people to the end (no swaps, no third person).
                identified = player_follow.follow_players(
                    parsed, 2, log, fps=job.video_meta.get("fps") or 30.0
                )
            except Exception as exc:  # noqa: BLE001 - labelling must never fail the job
                log.warning(f"player identification failed ({type(exc).__name__}: {exc}); "
                            "falling back to time/position linking", stage="skeleton")
                log.debug(traceback.format_exc(), stage="skeleton")
                identified = None
        if identified is None:
            parsed = skeleton_service.link_player_tracklets(
                parsed, 1 if job.sport == "swimming" else 2, log,
                fps=job.video_meta.get("fps") or 30.0,
            )
        # Preserve the raw OpenPose JSON; this is an additional artefact.
        skeleton_service.save_skeleton(parsed, job.skeleton_json, log)
        job.skeleton = skeleton_service.summarize(parsed)
        job.skeleton["skeleton_json_path"] = str(job.skeleton_json)
        job.skeleton["openpose_json_dir"] = str(job.openpose_json_dir)
        store.finish_stage(job, "skeleton", "done")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        log.error(f"skeleton extraction failed: {msg}", stage="skeleton")
        log.debug(traceback.format_exc(), stage="skeleton")
        store.finish_stage(job, "skeleton", "failed", msg)
        store.set_status(job, "failed", "skeleton", msg)
        _write_result(job)
        return

    # ---------------- stage: convert ----------------
    store.start_stage(job, "convert")
    try:
        # Swimming: one swimmer only (see swim_service for why).
        max_persons = 1 if job.sport == "swimming" else 2
        converted = skeleton_converter.convert(parsed, log, max_persons=max_persons)
        job.conversion = converted["meta"]
        job.converted_json.write_text(
            json.dumps(
                {
                    "meta": converted["meta"],
                    "pose_results": converted["pose_results"],
                }
            ),
            encoding="utf-8",
        )
        job.conversion["converted_json_path"] = str(job.converted_json)
        store.finish_stage(job, "convert", "done")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        log.error(f"skeleton conversion failed: {msg}", stage="convert")
        log.debug(traceback.format_exc(), stage="convert")
        store.finish_stage(job, "convert", "failed", msg)
        store.finish_stage(job, "mmaction", "skipped")
        # OpenPose still succeeded - report partial, not total failure.
        store.set_status(job, "partial", "convert", msg)
        _write_result(job)
        return

    # Players-only overlay ("Player 1" / "Player 2" on the tracks the model
    # will classify). Never fatal: on failure the engine's overlay remains.
    try:
        players = players_service.render(
            parsed,
            converted["meta"]["track_ids_used"],
            analysis_input,
            job.dir / "skeleton" / "players.json",
            job.openpose_dir / "players_overlay.avi",
            job.openpose_video_mp4,
            log,
        )
        job.openpose["players"] = players
        job.openpose["mp4_written"] = True
        job.openpose["overlay"] = "players only"
    except Exception as exc:  # noqa: BLE001
        log.warning(f"players overlay skipped: {exc}", stage="skeleton")

    # ---------------- stage: mmaction ----------------
    store.start_stage(job, "mmaction")
    try:
        h = job.video_meta["height"]
        w = job.video_meta["width"]
        fps = job.video_meta.get("fps") or 30.0

        if job.sport == "swimming":
            swim = swim_service.analyse_swimming(
                converted["pose_results"], (h, w), fps, log
            )
            dominant = swim["dominant_stroke"]
            pred = {
                "top1": {"label": dominant} if dominant else None,
                "topk": [
                    {"rank": i + 1, "label": d["stroke"], "score": d["share"]}
                    for i, d in enumerate(swim["distribution"])
                ],
                "swimming": swim,
                "analysis_mode": "swimming_sliding_window",
                "num_frames_sent": len(converted["pose_results"]),
                "num_persons_sent": converted["meta"]["num_persons"],
                "inference_s": swim["duration_s"],
                "device_used": s.resolved_device(),
                "device_requested": s.device,
                "dataset": swim["model"]["dataset"],
                "model_name": swim["model"]["name"],
                "domain_warning": swim["domain_warning"],
            }
        elif s.timeline_enabled:
            # Classify overlapping windows so every frame is analysed, then keep
            # the aggregate in the same shape the UI already understands.
            tl = timeline_service.analyse_timeline(
                converted["pose_results"], (h, w), log, fps=fps
            )
            pred = {
                "top1": tl["top1"],
                "topk": tl["overall_topk"],
                "timeline": tl,
                "analysis_mode": "sliding_window",
                "num_frames_sent": len(converted["pose_results"]),
                "num_persons_sent": converted["meta"]["num_persons"],
                "num_keypoints_sent": converted["meta"]["target_num_keypoints"],
                "graph_num_node": 17,
                "num_classes": None,
                "inference_s": tl["duration_s"],
                "device_used": s.resolved_device(),
                "device_requested": s.device,
                "dataset": s.model_dataset,
                "model_name": s.model_name,
                "domain_warning": (
                    f"This checkpoint was trained on {s.model_dataset}, which contains "
                    "no karate classes. Predictions are OUT-OF-DOMAIN."
                ),
            }
        else:
            pred = mmaction_service.run(converted["pose_results"], (h, w), log)
            pred["analysis_mode"] = "whole_clip"

        mmaction_service.save_prediction(pred, job.prediction_json, log)
        job.mmaction = pred
        job.mmaction["prediction_json_path"] = str(job.prediction_json)
        store.finish_stage(job, "mmaction", "done")
        store.finish_stage(job, "completed", "done")
        store.set_status(job, "completed", "completed")
        log.info("pipeline completed successfully")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        log.error(f"MMAction2 failed: {msg}", stage="mmaction")
        log.debug(traceback.format_exc(), stage="mmaction")
        store.finish_stage(job, "mmaction", "failed", msg)
        # Explicitly a partial success: OpenPose + skeleton are valid and usable.
        store.set_status(job, "partial", "mmaction", msg)
        log.warning(
            "OpenPose and skeleton extraction succeeded; only MMAction2 failed. "
            "The OpenPose outputs remain valid and downloadable."
        )

    _write_result(job)
