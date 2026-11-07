"""Long-lived MMAction2 inference worker for real-time use.

The batch worker (mmaction_infer.py) pays the model-load cost on every call,
which is fine for one-shot jobs but hopeless for streaming. This process loads
the recognizer ONCE and then answers repeated requests over stdin/stdout.

Protocol - newline-delimited JSON, one request per line:

    stdin  <- {"id": 7, "pose_results": [...], "img_shape": [h, w], "topk": 5}
    stdout -> {"id": 7, "ok": true, "top1": {...}, "topk": [...], "inference_s": 0.31}
    stdout -> {"id": 7, "ok": false, "error": "..."}

A first line is emitted once the model is warm:

    {"ready": true, "device": "cuda:0", "num_classes": 60, ...}

Startup config arrives as argv[1] holding a JSON blob so the caller never has to
quote long Windows paths through a shell.

REAL-TIME ACCURACY TRADEOFF
---------------------------
The shipped test_pipeline uses UniformSampleFrames(clip_len=100, num_clips=10),
i.e. it samples ten 100-frame clips and averages their scores. That is ~10x the
compute per prediction. For streaming we rebuild the pipeline with num_clips=1
(configurable via "num_clips"), which cuts latency roughly 10-fold at some cost
in temporal averaging. The value actually used is reported in every response so
the UI can state it rather than implying full-accuracy inference.
"""
from __future__ import annotations

import json
import sys
import time
import traceback


# mmengine/mmcv print to stdout during checkpoint loading (e.g.
# "Loads checkpoint by local backend from path: ..."), which would corrupt a
# bare newline-delimited-JSON protocol. Every protocol line therefore carries a
# unique prefix and the reader ignores anything else.
MSG = "@@MMA@@"


def _emit(obj: dict) -> None:
    sys.stdout.write(MSG + json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    cfg_blob = json.loads(sys.argv[1])
    config_path = cfg_blob["config"]
    checkpoint = cfg_blob["checkpoint"]
    label_map_path = cfg_blob["label_map"]
    device = cfg_blob.get("device", "cpu")
    num_clips = int(cfg_blob.get("num_clips", 1))

    import numpy as np
    import torch
    import mmengine
    from mmengine.dataset import Compose
    from mmaction.apis import inference_skeleton, init_recognizer
    from mmaction.registry import TRANSFORMS  # noqa: F401  (ensures registry is populated)

    effective_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        effective_device = "cpu"

    cfg = mmengine.Config.fromfile(config_path)
    model = init_recognizer(cfg, checkpoint, device=effective_device)

    with open(label_map_path, encoding="utf-8") as fh:
        labels = [ln.strip() for ln in fh if ln.strip()]

    # Rebuild the test pipeline with a reduced clip count for latency.
    #
    # DecompressPose is dropped: PoseC3D configs start with it because their
    # annotation .pkl stores poses in a packed form, but inference_skeleton
    # hands over plain 'keypoint'/'keypoint_score' arrays, and the transform
    # asserts on the missing packed fields. Removing it is what makes the
    # PoseC3D checkpoints (hmdb51/ucf101/k400) usable from live video at all.
    pipeline_cfg = [
        dict(t) for t in cfg.test_pipeline if t.get("type") != "DecompressPose"
    ]
    clip_len = None
    for step in pipeline_cfg:
        if step.get("type") == "UniformSampleFrames":
            step["num_clips"] = num_clips
            clip_len = step.get("clip_len")
    test_pipeline = Compose(pipeline_cfg)

    # GCN models expose backbone.graph.num_node; PoseC3D (Recognizer3D) does
    # not - it takes heatmaps, and its joint count is cls_head in_channels.
    expected_nodes = getattr(getattr(model, "backbone", None), "graph", None)
    expected_nodes = getattr(expected_nodes, "num_node", None)
    if expected_nodes is None:
        try:
            expected_nodes = cfg.model["backbone"].get("in_channels")
        except (KeyError, TypeError, AttributeError):
            expected_nodes = None

    _emit(
        {
            "ready": True,
            "device_requested": device,
            "device_used": effective_device,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_classes": len(labels),
            "graph_num_node": expected_nodes,
            "num_clips": num_clips,
            "clip_len": clip_len,
            "model_name": cfg.model.get("type", "unknown"),
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            if req.get("cmd") == "shutdown":
                break

            topk = int(req.get("topk", 5))
            img_shape = tuple(req["img_shape"])
            pose_results = [
                {
                    "keypoints": np.array(f["keypoints"], dtype=np.float32),
                    "keypoint_scores": np.array(f["keypoint_scores"], dtype=np.float32),
                }
                for f in req["pose_results"]
            ]
            if not pose_results:
                raise ValueError("empty pose_results")

            v = pose_results[0]["keypoints"].shape[1]
            if expected_nodes is not None and v != expected_nodes:
                raise ValueError(
                    f"keypoint mismatch: graph expects {expected_nodes}, got {v}"
                )

            t0 = time.perf_counter()
            result = inference_skeleton(model, pose_results, img_shape, test_pipeline)
            dt = round(time.perf_counter() - t0, 3)

            scores = result.pred_score.detach().cpu().numpy().astype(float)
            order = np.argsort(scores)[::-1][:topk]
            top = [
                {
                    "rank": i + 1,
                    "class_index": int(idx),
                    "label": labels[idx] if idx < len(labels) else f"class_{idx}",
                    "score": round(float(scores[idx]), 6),
                }
                for i, idx in enumerate(order)
            ]
            out = {
                "id": req_id,
                "ok": True,
                "top1": top[0] if top else None,
                "topk": top,
                "inference_s": dt,
                "num_frames_sent": len(pose_results),
                "num_persons_sent": int(pose_results[0]["keypoints"].shape[0]),
                "num_clips": num_clips,
                "device_used": effective_device,
            }
            # Full score vector, needed to average across sliding windows.
            # Averaging top-k alone would be wrong: different windows return
            # different label sets, so the means would not be comparable.
            if req.get("return_scores"):
                out["scores"] = [round(float(v), 6) for v in scores]
            _emit(out)
        except Exception as exc:  # noqa: BLE001 - report, never die
            _emit(
                {
                    "id": req_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:],
                }
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
