"""HTTP API for the analysis pipeline."""
from __future__ import annotations

import json
import shutil

import asyncio

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse

from ..config import get_settings
from ..jobs import store
from ..pipelines.video_analysis import analyze_video
from ..services import mmaction_service, openpose_service, pose_engine_service, realtime_service
from ..services.keypoints import conversion_table, dropped_table
from ..services.video_service import ALLOWED_SUFFIXES
from ..utils.logging_utils import JobLogger, get_logs

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    s = get_settings()
    report = s.environment_report()
    op_problems = openpose_service.preflight()
    mm_problems = mmaction_service.preflight()
    return {
        "status": "ok",
        "openpose_ready": not op_problems,
        "openpose_problems": op_problems,
        "mmaction_ready": not mm_problems,
        "mmaction_problems": mm_problems,
        "environment": report,
        "openpose_version": openpose_service.openpose_version(),
        "pose_engines": [
            {"id": k, **v, "ready": s.pose_engine_ready(k)}
            for k, v in pose_engine_service.ENGINES.items()
        ],
        "swimming": {
            "model_ready": s.swim_model_ready,
            "config": str(s.swim_config),
            "checkpoint": str(s.swim_checkpoint),
            "label_map": str(s.swim_label_map),
            "window_frames": s.swim_window,
            "stride_frames": s.swim_stride,
            "min_confidence": s.swim_min_confidence,
        },
    }


@router.get("/keypoint-mapping")
def keypoint_mapping() -> dict:
    """The documented BODY_25 -> COCO-17 conversion, exposed for transparency."""
    s = get_settings()
    return {
        "source": {"format": "BODY_25", "num_keypoints": 25, "producer": "OpenPose 1.7.0"},
        "target": {
            "format": "COCO-17",
            "num_keypoints": 17,
            "layout": s.model_keypoint_layout,
            "required_by": f"{s.model_name} / {s.model_dataset}",
        },
        "mapping": conversion_table(),
        "dropped_body25_joints": dropped_table(),
        "method": "pure gather (subset + reorder); no interpolated or synthesised joints",
    }


@router.post("/analyze")
async def analyze(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    sport: str = Form("karate"),
    pose_engine: str = Form("openpose"),
) -> dict:
    s = get_settings()
    if sport not in ("karate", "swimming"):
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'")
    if pose_engine not in pose_engine_service.ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown pose engine '{pose_engine}'")
    if not s.pose_engine_ready(pose_engine):
        raise HTTPException(status_code=503, detail=f"Pose engine '{pose_engine}' is not installed")
    if sport == "swimming" and not s.swim_model_ready:
        raise HTTPException(
            status_code=503,
            detail="The swimming model is not trained/installed yet (models/swim/).",
        )
    live = realtime_service.get_session()
    if live is not None and live.snapshot().get("running"):
        raise HTTPException(
            status_code=409,
            detail=f"Live detection ({live.sport}) is using the camera and GPU - "
            "stop it before analysing a video.",
        )
    filename = file.filename or "upload.mp4"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    job = store.create(filename=filename, size_bytes=0, sport=sport, pose_engine=pose_engine)
    log = JobLogger(job.job_id, job.dir)
    log.info(f"upload started: {filename}", stage="upload")
    store.start_stage(job, "upload")

    try:
        with job.input_video.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
    except OSError as exc:
        msg = f"Could not save upload: {exc}"
        log.error(msg, stage="upload")
        store.finish_stage(job, "upload", "failed", msg)
        store.set_status(job, "failed", "upload", msg)
        raise HTTPException(status_code=500, detail=msg) from exc
    finally:
        await file.close()

    size = job.input_video.stat().st_size
    job.size_bytes = size
    if size == 0:
        msg = "Uploaded file is empty."
        log.error(msg, stage="upload")
        store.finish_stage(job, "upload", "failed", msg)
        store.set_status(job, "failed", "upload", msg)
        raise HTTPException(status_code=400, detail=msg)
    if size > s.max_upload_mb * 1024 * 1024:
        msg = f"File is {size / 1e6:.1f} MB, over the {s.max_upload_mb} MB limit."
        log.error(msg, stage="upload")
        store.finish_stage(job, "upload", "failed", msg)
        store.set_status(job, "failed", "upload", msg)
        raise HTTPException(status_code=413, detail=msg)

    log.info(f"upload completed: {size / 1e6:.2f} MB -> {job.input_video.name}", stage="upload")
    store.finish_stage(job, "upload", "done")
    store.set_status(job, "queued", "validate")

    background.add_task(analyze_video, job)
    return {"job_id": job.job_id, "status": "queued", "filename": filename, "size_bytes": size}


def _get_job_or_404(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return job


@router.get("/jobs")
def list_jobs(sport: str | None = None) -> dict:
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "filename": j.filename,
                "sport": j.sport,
                "pose_engine": j.pose_engine,
                "status": j.status,
                "stage": j.stage,
                "created_at": j.created_at,
            }
            for j in store.list()
            if sport is None or j.sport == sport
        ]
    }


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "filename": job.filename,
        "sport": job.sport,
        "pose_engine": job.pose_engine,
        "size_bytes": job.size_bytes,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "stages": [
            {
                "name": job.stages[k].name,
                "state": job.stages[k].state,
                "duration_s": job.stages[k].duration_s,
                "error": job.stages[k].error,
            }
            for k in job.stages
        ],
    }


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    s = get_settings()
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "filename": job.filename,
        "sport": job.sport,
        "pose_engine": job.pose_engine,
        "size_bytes": job.size_bytes,
        "error": job.error,
        "video_meta": job.video_meta,
        "openpose": job.openpose,
        "skeleton": job.skeleton,
        "conversion": job.conversion,
        "mmaction": job.mmaction,
        "model_info": {
            "model_name": "ST-GCN",
            "dataset": "SwimXYZ (synthetic)",
            "keypoint_layout": "coco",
            "config": str(s.swim_config),
            "checkpoint": str(s.swim_checkpoint),
            "status_banner": "SYNTHETIC-TRAINED SWIMMING MODEL",
            "disclaimer": (
                "Fine-tuned on synthetic SwimXYZ swimmers. Accuracy on real pool "
                "footage has not been measured yet."
            ),
        } if job.sport == "swimming" else {
            "model_name": s.model_name,
            "dataset": s.model_dataset,
            "keypoint_layout": s.model_keypoint_layout,
            "config": str(s.mmaction_config),
            "checkpoint": str(s.checkpoint_path),
            "status_banner": "OUT-OF-DOMAIN KARATE TEST",
            "disclaimer": (
                f"{s.model_name} here is pretrained on {s.model_dataset}. That dataset "
                "contains no karate classes, so predictions are NOT karate technique "
                "labels. This run only validates the pipeline."
            ),
        },
        "stages": [
            {
                "name": job.stages[k].name,
                "state": job.stages[k].state,
                "duration_s": job.stages[k].duration_s,
                "error": job.stages[k].error,
            }
            for k in job.stages
        ],
        "outputs": _outputs(job),
    }


def _outputs(job) -> dict:
    """Which artefacts actually exist on disk, with sizes."""
    items = {
        "input_video": job.input_video,
        "openpose_video_avi": job.openpose_video_avi,
        "openpose_video_mp4": job.openpose_video_mp4,
        "openpose_json_dir": job.openpose_json_dir,
        "skeleton_json": job.skeleton_json,
        "skeleton_coco17_json": job.converted_json,
        "prediction_json": job.prediction_json,
        "result_json": job.result_json,
        "job_log": job.dir / "job.log",
    }
    out = {}
    for name, path in items.items():
        if path.is_dir():
            files = list(path.glob("*"))
            out[name] = {
                "path": str(path),
                "exists": True,
                "type": "directory",
                "file_count": len(files),
                "size_bytes": sum(f.stat().st_size for f in files if f.is_file()),
            }
        else:
            exists = path.exists()
            out[name] = {
                "path": str(path),
                "exists": exists,
                "type": "file",
                "size_bytes": path.stat().st_size if exists else 0,
            }
    return out


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str) -> dict:
    _get_job_or_404(job_id)
    return {"job_id": job_id, "logs": get_logs(job_id)}


@router.get("/jobs/{job_id}/skeleton")
def job_skeleton(job_id: str, full: bool = False) -> JSONResponse:
    job = _get_job_or_404(job_id)
    if full:
        if not job.skeleton_json.exists():
            raise HTTPException(status_code=404, detail="skeleton.json not generated yet")
        return JSONResponse(json.loads(job.skeleton_json.read_text(encoding="utf-8")))
    if job.skeleton is None:
        raise HTTPException(status_code=404, detail="Skeleton not extracted yet")
    return JSONResponse(
        {
            "job_id": job_id,
            "summary": job.skeleton,
            "conversion": job.conversion,
            "hint": "add ?full=true for the complete per-frame BODY_25 JSON",
        }
    )


@router.get("/jobs/{job_id}/predictions")
def job_predictions(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    s = get_settings()
    if job.mmaction is None:
        stage = job.stages["mmaction"]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No predictions available (mmaction stage is '{stage.state}')."
                + (f" Error: {stage.error}" if stage.error else "")
            ),
        )
    return {
        "job_id": job_id,
        "model_name": s.model_name,
        "dataset": s.model_dataset,
        "status_banner": "OUT-OF-DOMAIN KARATE TEST",
        **job.mmaction,
    }


@router.get("/jobs/{job_id}/openpose-video")
def openpose_video(job_id: str):
    job = _get_job_or_404(job_id)
    if job.openpose_video_mp4.exists():
        return FileResponse(job.openpose_video_mp4, media_type="video/mp4",
                            filename=f"{job_id}_openpose.mp4")
    if job.openpose_video_avi.exists():
        # Served for download; most browsers will not play MJPG/AVI inline.
        return FileResponse(job.openpose_video_avi, media_type="video/x-msvideo",
                            filename=f"{job_id}_openpose.avi")
    raise HTTPException(status_code=404, detail="OpenPose overlay video not generated")


# ----------------------------------------------------------------- real-time --
# The camera is opened by the OpenPose process on the backend host. The browser
# must NOT also call getUserMedia on the same device: a webcam can normally only
# be held by one process, so the live view is fed by OpenPose's rendered frames.


def _busy_batch_job():
    return next((j for j in store.list() if j.status in ("queued", "running")), None)


@router.post("/live/start")
def live_start(sport: str = "karate", standing: bool = True, engine: str = "openpose") -> dict:
    s = get_settings()
    if sport not in ("karate", "swimming"):
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'")
    if engine not in pose_engine_service.ENGINES or not s.pose_engine_ready(engine):
        raise HTTPException(status_code=400, detail=f"Pose engine '{engine}' is not available")
    # Two OpenPose processes on a 4 GB GPU slow BOTH ~10x (measured) - refuse
    # rather than let the live view look broken.
    busy = _busy_batch_job()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"A video analysis is running ({busy.filename}). Live detection "
            "shares the GPU with it - wait until it finishes.",
        )
    options = {"standing": standing} if sport == "swimming" else {}
    options["engine"] = engine
    try:
        session = realtime_service.start_session(sport, options)
    except realtime_service.SessionStarting as exc:
        # 409, not 500: a second click during the ~20s warm-up is a normal race,
        # and spawning another OpenPose would fight the first for the camera.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    snap = session.snapshot()
    return {
        "running": snap["running"],
        "openpose_device": snap["openpose_device"],
        "mmaction_device": snap["mmaction_device"],
        "window": snap["window"],
        "gpu_build": s.gpu_openpose_available,
        "warning": None
        if s.gpu_openpose_available
        else (
            "Running the CPU-only OpenPose build at roughly 0.45 fps. This is a live "
            "stream but NOT real-time; install the GPU build for real-time rates."
        ),
    }


@router.post("/live/stop")
def live_stop() -> dict:
    realtime_service.stop_session()
    return {"running": False}


@router.get("/live/status")
def live_status() -> dict:
    session = realtime_service.get_session()
    if session is None:
        return {"running": False}
    snap = session.snapshot()
    snap.pop("latest_frame_jpeg", None)  # too large for a status poll
    return snap


@router.websocket("/live/ws")
async def live_ws(ws: WebSocket) -> None:
    await ws.accept()
    last_frames = -1
    try:
        while True:
            session = realtime_service.get_session()
            if session is None:
                await ws.send_json({"running": False, "error": "no active session"})
                await asyncio.sleep(0.5)
                continue
            snap = session.snapshot()
            # Only ship the JPEG when the frame actually advanced, so an idle
            # camera does not saturate the socket with duplicates.
            if snap.get("frames_processed") == last_frames:
                snap.pop("latest_frame_jpeg", None)
            else:
                last_frames = snap.get("frames_processed", -1)
            await ws.send_json(snap)
            await asyncio.sleep(0.08)  # ~12 Hz UI refresh
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - never let a socket error kill the server
        try:
            await ws.close()
        except RuntimeError:
            pass


@router.get("/jobs/{job_id}/input-video")
def input_video(job_id: str):
    job = _get_job_or_404(job_id)
    if not job.input_video.exists():
        raise HTTPException(status_code=404, detail="Input video missing")
    return FileResponse(job.input_video, media_type="video/mp4",
                        filename=f"{job_id}_input.mp4")
