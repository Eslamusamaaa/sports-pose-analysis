"""Per-job logging.

Every job gets its own log file under data/jobs/<job_id>/job.log AND an
in-memory ring the API can hand to the browser, so the UI can show real backend
errors instead of a generic failure message.
"""
from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_BUFFERS: dict[str, list[dict]] = {}

LEVELS = {"debug", "info", "warning", "error"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobLogger:
    """Writes to data/jobs/<id>/job.log, stdout, and an in-memory buffer."""

    def __init__(self, job_id: str, job_dir: Path) -> None:
        self.job_id = job_id
        self.log_path = job_dir / "job.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _BUFFERS.setdefault(job_id, [])

    def _emit(self, level: str, message: str, stage: str | None = None) -> None:
        entry = {"ts": _now(), "level": level, "stage": stage, "message": message}
        with _LOCK:
            _BUFFERS.setdefault(self.job_id, []).append(entry)
        line = f"[{entry['ts']}] {level.upper():<7} {('['+stage+'] ') if stage else ''}{message}"
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        stream = sys.stderr if level == "error" else sys.stdout
        print(f"{self.job_id[:8]} {line}", file=stream, flush=True)

    def debug(self, message: str, stage: str | None = None) -> None:
        self._emit("debug", message, stage)

    def info(self, message: str, stage: str | None = None) -> None:
        self._emit("info", message, stage)

    def warning(self, message: str, stage: str | None = None) -> None:
        self._emit("warning", message, stage)

    def error(self, message: str, stage: str | None = None) -> None:
        self._emit("error", message, stage)


def get_logs(job_id: str) -> list[dict]:
    with _LOCK:
        return list(_BUFFERS.get(job_id, []))


def configure_root_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
