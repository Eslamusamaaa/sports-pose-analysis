"""Job registry and the per-job on-disk layout.

Deliberately an in-process store: this is a PoC, and a single uvicorn worker is
the documented way to run it. State is also mirrored to data/jobs/<id>/job.json
so a job survives inspection after the server stops.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .config import get_settings

Status = Literal["queued", "running", "completed", "failed", "partial"]

# Ordered pipeline stages, mirrored by the frontend status tracker.
STAGES: tuple[str, ...] = (
    "upload",
    "validate",
    "openpose",
    "skeleton",
    "convert",
    "mmaction",
    "completed",
)

StageState = Literal["pending", "running", "done", "failed", "skipped"]


@dataclass
class StageInfo:
    name: str
    state: StageState = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Job:
    job_id: str
    filename: str
    size_bytes: int
    sport: str = "karate"  # "karate" | "swimming" - selects model + UI page
    pose_engine: str = "openpose"  # "openpose" | "rtmpose" | "yolo11" - any sport
    status: Status = "queued"
    stage: str = "upload"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    error: str | None = None
    stages: dict[str, StageInfo] = field(
        default_factory=lambda: {s: StageInfo(name=s) for s in STAGES}
    )
    # Populated by the pipeline as each stage completes.
    video_meta: dict | None = None
    openpose: dict | None = None
    skeleton: dict | None = None
    conversion: dict | None = None
    mmaction: dict | None = None

    # --- paths --------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return get_settings().jobs_dir / self.job_id

    @property
    def input_video(self) -> Path:
        return self.dir / "input.mp4"

    @property
    def openpose_dir(self) -> Path:
        return self.dir / "openpose"

    @property
    def openpose_json_dir(self) -> Path:
        return self.openpose_dir / "json"

    @property
    def openpose_video_avi(self) -> Path:
        return self.openpose_dir / "output_video.avi"

    @property
    def openpose_video_mp4(self) -> Path:
        """Browser-playable transcode. AVI/MJPG will not play in a <video> tag."""
        return self.openpose_dir / "output_video.mp4"

    @property
    def skeleton_json(self) -> Path:
        return self.dir / "skeleton" / "skeleton.json"

    @property
    def converted_json(self) -> Path:
        return self.dir / "skeleton" / "skeleton_coco17.json"

    @property
    def prediction_json(self) -> Path:
        return self.dir / "mmaction" / "prediction.json"

    @property
    def result_json(self) -> Path:
        return self.dir / "result.json"

    def ensure_dirs(self) -> None:
        for p in (
            self.dir,
            self.openpose_dir,
            self.openpose_json_dir,
            self.dir / "skeleton",
            self.dir / "mmaction",
        ):
            p.mkdir(parents=True, exist_ok=True)

    # --- serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__ if f != "stages"}
        job = cls(**{k: v for k, v in d.items() if k in known})
        for s in d.get("stages") or []:
            if s.get("name") in job.stages:
                job.stages[s["name"]] = StageInfo(**s)
        return job

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [asdict(self.stages[s]) for s in STAGES]
        return d

    def persist(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.result_json.parent.mkdir(parents=True, exist_ok=True)
            (self.dir / "job.json").write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            pass


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Rehydrate jobs persisted by earlier server runs so they stay browsable.

        A job still queued/running when the server stopped can never resume, so
        it is reported as failed rather than left spinning forever in the UI.
        """
        jobs_dir = get_settings().jobs_dir
        if not jobs_dir.is_dir():
            return
        for path in jobs_dir.glob("*/job.json"):
            try:
                job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error = job.error or "Interrupted: server stopped before the job finished"
            self._jobs[job.job_id] = job

    def create(self, filename: str, size_bytes: int, sport: str = "karate",
               pose_engine: str = "openpose") -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, filename=filename, size_bytes=size_bytes, sport=sport,
                  pose_engine=pose_engine)
        job.ensure_dirs()
        with self._lock:
            self._jobs[job_id] = job
        job.persist()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # --- mutation -----------------------------------------------------------
    def start_stage(self, job: Job, stage: str) -> None:
        with self._lock:
            job.stage = stage
            job.status = "running"
            info = job.stages[stage]
            info.state = "running"
            info.started_at = _now()
            job.updated_at = _now()
        job.persist()

    def finish_stage(self, job: Job, stage: str, state: StageState = "done",
                     error: str | None = None) -> None:
        with self._lock:
            info = job.stages[stage]
            info.state = state
            info.finished_at = _now()
            info.error = error
            if info.started_at:
                t0 = datetime.fromisoformat(info.started_at)
                t1 = datetime.fromisoformat(info.finished_at)
                info.duration_s = round((t1 - t0).total_seconds(), 2)
            job.updated_at = _now()
        job.persist()

    def set_status(self, job: Job, status: Status, stage: str | None = None,
                   error: str | None = None) -> None:
        with self._lock:
            job.status = status
            if stage:
                job.stage = stage
            if error:
                job.error = error
            job.updated_at = _now()
        job.persist()


store = JobStore()
