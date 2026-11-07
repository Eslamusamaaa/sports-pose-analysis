"""Invokes the MMAction2 inference worker in its isolated venv."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger

RESULT_PREFIX = "MMACTION_RESULT_JSON:"
WORKER = BACKEND_DIR / "mmaction_infer.py"


class MMActionError(RuntimeError):
    pass


def preflight() -> list[str]:
    s = get_settings()
    problems: list[str] = []
    if not s.active_mmaction_python.exists():
        problems.append(
            f"MMAction2 venv python not found: {s.active_mmaction_python} "
            "(run scripts/setup_mmaction_env.ps1)"
        )
    if not s.mmaction_config.exists():
        problems.append(f"MMAction2 config not found: {s.mmaction_config}")
    if not s.checkpoint_path.exists():
        problems.append(f"Checkpoint not found: {s.checkpoint_path}")
    if not s.label_map_path.exists():
        problems.append(f"Label map not found: {s.label_map_path}")
    if not WORKER.exists():
        problems.append(f"Inference worker not found: {WORKER}")
    return problems


def run(pose_results: list[dict], img_shape: tuple[int, int], log: JobLogger,
        topk: int = 5) -> dict:
    s = get_settings()
    problems = preflight()
    if problems:
        raise MMActionError("; ".join(problems))

    payload = {
        "pose_results": pose_results,
        "img_shape": [int(img_shape[0]), int(img_shape[1])],
        "config": str(s.mmaction_config),
        "checkpoint": str(s.checkpoint_path),
        "label_map": str(s.label_map_path),
        "device": s.device,
        "topk": topk,
    }
    cmd = [str(s.active_mmaction_python), str(WORKER)]
    log.info(f"MMAction2 command: {' '.join(cmd)}", stage="mmaction")
    log.info(
        f"MMAction2 started: {len(pose_results)} frames, "
        f"device requested={s.device}, config={s.mmaction_config.name}",
        stage="mmaction",
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=s.mmaction_timeout_s,
            cwd=str(s.mmaction2_dir),  # so relative config _base_ paths resolve
        )
    except subprocess.TimeoutExpired as exc:
        raise MMActionError(f"MMAction2 timed out after {s.mmaction_timeout_s}s") from exc
    elapsed = round(time.perf_counter() - t0, 2)

    stderr_tail = (proc.stderr or "").strip()[-4000:]
    if proc.returncode != 0:
        log.error(f"MMAction2 exited {proc.returncode}", stage="mmaction")
        raise MMActionError(
            f"MMAction2 failed (exit {proc.returncode}).\n--- stderr ---\n"
            f"{stderr_tail or '(empty)'}"
        )

    line = next(
        (ln for ln in (proc.stdout or "").splitlines() if ln.startswith(RESULT_PREFIX)),
        None,
    )
    if line is None:
        raise MMActionError(
            "MMAction2 worker produced no result line.\n"
            f"--- stdout ---\n{(proc.stdout or '')[-1500:]}\n"
            f"--- stderr ---\n{stderr_tail or '(empty)'}"
        )

    result = json.loads(line[len(RESULT_PREFIX):])
    result["subprocess_duration_s"] = elapsed
    result["dataset"] = s.model_dataset
    result["stderr_tail"] = stderr_tail[-800:]
    # Never let a prediction be shown without its provenance.
    result["domain_warning"] = (
        f"This checkpoint was trained on {s.model_dataset}, which contains no "
        "karate classes. Predictions on karate footage are OUT-OF-DOMAIN and "
        "must not be read as karate technique labels."
    )

    top1 = result.get("top1") or {}
    log.info(
        f"MMAction2 completed in {elapsed}s "
        f"(load {result.get('model_load_s')}s, infer {result.get('inference_s')}s) "
        f"on device={result.get('device_used')}",
        stage="mmaction",
    )
    log.info(
        f"prediction: top1='{top1.get('label')}' score={top1.get('score')} "
        f"(class {top1.get('class_index')} of {result.get('num_classes')})",
        stage="mmaction",
    )
    if result.get("note"):
        log.warning(f"MMAction2 note: {result['note']}", stage="mmaction")
    return result


def save_prediction(result: dict, dest: Path, log: JobLogger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(f"prediction written: {dest.name}", stage="mmaction")
