"""FastAPI entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000

Single worker on purpose: the job registry is in-process (see jobs.py).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import get_settings
from .utils.logging_utils import configure_root_logging

configure_root_logging()

app = FastAPI(
    title="Karate AI Web - OpenPose + MMAction2 PoC",
    version="0.1.0",
    description=(
        "Proof of concept: browser upload -> OpenPose BODY_25 -> skeleton -> "
        "BODY_25-to-COCO17 conversion -> MMAction2 ST-GCN -> results in the browser."
    ),
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup_report() -> None:
    """Log what is and is not available, so misconfiguration is obvious at boot."""
    from .services import mmaction_service, openpose_service, realtime_service

    s = get_settings()
    s.jobs_dir.mkdir(parents=True, exist_ok=True)
    # An orphan from a previous run holds the camera and ~2 GB of VRAM.
    orphans = realtime_service.kill_orphaned_openpose()
    print("=" * 72, flush=True)
    print("karate-ai-web backend starting", flush=True)
    if orphans:
        print(f"  reaped {orphans} orphaned OpenPose process(es)", flush=True)
    print(f"  data dir       : {s.data_dir}", flush=True)
    print(f"  openpose bin   : {s.openpose_bin}", flush=True)
    print(f"  openpose models: {s.openpose_model_dir}", flush=True)
    print(f"  mmaction2 repo : {s.mmaction2_dir}", flush=True)
    print(f"  mmaction python: {s.mmaction_python}", flush=True)
    print(f"  checkpoint     : {s.checkpoint_path.name}", flush=True)
    print(f"  device request : {s.device}", flush=True)
    op = openpose_service.preflight()
    mm = mmaction_service.preflight()
    print(f"  OpenPose ready : {not op}" + (f" -> {op}" if op else ""), flush=True)
    print(f"  MMAction2 ready: {not mm}" + (f" -> {mm}" if mm else ""), flush=True)
    print("=" * 72, flush=True)


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    """Release the camera and GPU on a clean exit."""
    from .services import realtime_service

    try:
        realtime_service.stop_session()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        pass


@app.get("/")
def root() -> dict:
    return {
        "name": "karate-ai-web backend",
        "docs": "/docs",
        "health": "/api/health",
    }
