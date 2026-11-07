"""Runs several skeleton models over the same clip and pools their evidence.

See model_zoo for why the models cannot simply be score-averaged. The pooling
rule implemented here:

  1. Each model classifies the same window and returns its full score vector.
  2. Only classes mapped to a shared CONCEPT contribute.
  3. A score is converted to a "lift" over that model's uniform baseline:

         lift = score / (1 / num_classes)

     so 0.10 from a 400-class model (lift 40) outweighs 0.10 from a 51-class
     model (lift 5.1). Without this, the coarse models would dominate purely
     because their probability mass is spread over fewer classes.

  4. Lifts for the same concept are summed across models, and the number of
     models that fired above threshold is recorded as `votes`.

`votes` matters more than the pooled number: two independent architectures
trained on different datasets agreeing on "kick" in the same second is far
stronger evidence than one model being confident alone.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger
from .model_zoo import (
    CONCEPT_AR,
    CONCEPT_ORDER,
    DEFAULT_ENSEMBLE,
    ModelSpec,
    available_models,
    resolve_concept_indices,
)

WORKER = BACKEND_DIR / "mmaction_worker.py"
MSG_PREFIX = "@@MMA@@"

# A concept only counts as a vote above this lift. 2.0 = twice what the class
# would get by chance; low enough to catch weak out-of-domain signals, high
# enough to exclude noise.
VOTE_LIFT_THRESHOLD = 2.0


class EnsembleError(RuntimeError):
    pass


class ModelWorker:
    """One persistent recognizer process for one checkpoint."""

    def __init__(self, spec: ModelSpec, log: JobLogger, num_clips: int = 1) -> None:
        s = get_settings()
        self.spec = spec
        self.log = log
        self.concept_idx = resolve_concept_indices(spec)
        if not self.concept_idx:
            log.warning(
                f"{spec.key}: no concept classes resolved - label map may not "
                "match the checkpoint",
                stage="ensemble",
            )
        cfg = {
            "config": str(spec.config),
            "checkpoint": str(spec.checkpoint),
            "label_map": str(spec.label_map),
            "device": s.device,
            "num_clips": num_clips,
        }
        self.proc = subprocess.Popen(
            [str(s.active_mmaction_python), str(WORKER), json.dumps(cfg)],
            cwd=str(s.mmaction2_dir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready = self._read(timeout=420)
        if not ready or not ready.get("ready"):
            err = ""
            try:
                err = (self.proc.stderr.read() or "")[-800:]
            except OSError:
                pass
            raise EnsembleError(f"{spec.key} failed to load. {err}")
        self.ready = ready
        self.num_classes = ready.get("num_classes") or spec.num_classes
        self.baseline = 1.0 / max(self.num_classes, 1)
        log.info(
            f"loaded {spec.key} ({spec.name} / {spec.dataset}, "
            f"{self.num_classes} classes, clip_len={ready.get('clip_len')}) "
            f"on {ready.get('device_used')}",
            stage="ensemble",
        )

    def _read(self, timeout: float = 300.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                return None
            if line.startswith(MSG_PREFIX):
                return json.loads(line[len(MSG_PREFIX):])
        return None

    def classify(self, pose_results, img_shape, req_id: int) -> dict | None:
        req = {
            "id": req_id, "pose_results": pose_results,
            "img_shape": [int(img_shape[0]), int(img_shape[1])],
            "topk": 5, "return_scores": True,
        }
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EnsembleError(f"{self.spec.key} pipe closed: {exc}") from exc
        return self._read(timeout=300)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
            except OSError:
                pass


def _pool(per_model: list[dict]) -> dict:
    """Combine one window's per-model results into concept evidence."""
    pooled: dict[str, float] = defaultdict(float)
    votes: dict[str, list[str]] = defaultdict(list)
    for entry in per_model:
        for concept, lift in entry["concepts"].items():
            pooled[concept] += lift
            if lift >= VOTE_LIFT_THRESHOLD:
                votes[concept].append(entry["dataset"])
    ranked = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "concepts": [
            {
                "concept": c,
                "concept_ar": CONCEPT_AR.get(c, c),
                "pooled_lift": round(v, 2),
                "votes": len(votes.get(c, [])),
                "voted_by": votes.get(c, []),
            }
            for c, v in ranked
            if v > 0
        ],
        "top_concept": ranked[0][0] if ranked else None,
    }


def analyse_ensemble(
    pose_results: list[dict],
    img_shape: tuple[int, int],
    log: JobLogger,
    fps: float = 30.0,
    window: int | None = None,
    stride: int | None = None,
    model_keys: list[str] | None = None,
) -> dict:
    s = get_settings()
    window = window or s.timeline_window
    stride = stride or s.timeline_stride
    total = len(pose_results)
    if total == 0:
        raise EnsembleError("no frames to analyse")

    if model_keys is None:
        configured = [k.strip() for k in s.ensemble_models.split(",") if k.strip()]
        model_keys = configured or list(DEFAULT_ENSEMBLE)
    specs = [m for m in available_models() if m.key in model_keys]
    if not specs:
        raise EnsembleError("no models available - run scripts/download_models.ps1")

    starts = list(range(0, max(total - window, 0) + 1, stride)) or [0]
    last = max(total - window, 0)
    if starts[-1] != last:
        starts.append(last)

    log.info(
        f"ensemble: {len(specs)} models x {len(starts)} windows "
        f"(window={window}, stride={stride})",
        stage="ensemble",
    )

    workers: list[ModelWorker] = []
    t0 = time.perf_counter()
    try:
        for spec in specs:
            try:
                workers.append(ModelWorker(spec, log, num_clips=s.realtime_num_clips))
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill the run
                log.error(f"skipping {spec.key}: {exc}", stage="ensemble")

        if not workers:
            raise EnsembleError("every model failed to load")

        windows: list[dict] = []
        concept_totals: dict[str, float] = defaultdict(float)

        for i, start in enumerate(starts):
            chunk = pose_results[start:start + window]
            if not chunk:
                continue
            per_model = []
            for w in workers:
                resp = w.classify(chunk, img_shape, req_id=i)
                if not resp or not resp.get("ok"):
                    continue
                scores = resp.get("scores") or []
                concepts: dict[str, float] = defaultdict(float)
                for idx, concept in w.concept_idx.items():
                    if idx < len(scores):
                        concepts[concept] += scores[idx] / w.baseline
                per_model.append(
                    {
                        "key": w.spec.key,
                        "dataset": w.spec.dataset,
                        "top1": resp.get("top1"),
                        "concepts": dict(concepts),
                    }
                )
            pooled = _pool(per_model)
            for c in pooled["concepts"]:
                concept_totals[c["concept"]] += c["pooled_lift"]

            end = start + len(chunk) - 1
            windows.append(
                {
                    "index": i,
                    "start_frame": start,
                    "end_frame": end,
                    "start_s": round(start / fps, 2) if fps else None,
                    "end_s": round((end + 1) / fps, 2) if fps else None,
                    "per_model": per_model,
                    **pooled,
                }
            )
            if (i + 1) % 25 == 0 or i + 1 == len(starts):
                log.info(f"  {i + 1}/{len(starts)} windows", stage="ensemble")
    finally:
        for w in workers:
            w.close()

    elapsed = round(time.perf_counter() - t0, 2)
    n = max(len(windows), 1)

    # Windows where at least two different datasets agree on a striking concept.
    strong = [
        w for w in windows
        if any(c["votes"] >= 2 and c["concept"] in ("punch", "kick") for c in w["concepts"])
    ]

    log.info(
        f"ensemble completed: {len(windows)} windows in {elapsed}s; "
        f"{len(strong)} window(s) with >=2 models agreeing on a strike",
        stage="ensemble",
    )

    return {
        "windows": windows,
        "num_windows": len(windows),
        "window_frames": window,
        "stride_frames": stride,
        "window_seconds": round(window / fps, 2) if fps else None,
        "frames_covered": total,
        "duration_s": elapsed,
        "models": [
            {
                "key": w.spec.key, "name": w.spec.name, "dataset": w.spec.dataset,
                "num_classes": w.num_classes,
            }
            for w in workers
        ],
        "concept_order": CONCEPT_ORDER,
        "concept_totals": [
            {"concept": c, "concept_ar": CONCEPT_AR.get(c, c), "mean_lift": round(v / n, 2)}
            for c, v in sorted(concept_totals.items(), key=lambda kv: kv[1], reverse=True)
        ],
        "strike_windows": [
            {
                "start_s": w["start_s"], "end_s": w["end_s"],
                "concepts": [c for c in w["concepts"] if c["votes"] >= 2],
                "per_model_top1": [
                    {"dataset": m["dataset"], "label": (m["top1"] or {}).get("label"),
                     "score": (m["top1"] or {}).get("score")}
                    for m in w["per_model"]
                ],
            }
            for w in strong
        ],
        "vote_threshold_lift": VOTE_LIFT_THRESHOLD,
    }
