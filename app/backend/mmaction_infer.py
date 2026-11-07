"""Standalone MMAction2 inference worker.

Runs INSIDE third_party/mmaction-venv (Python 3.11 + torch 2.1.0 + mmcv 2.1.0),
not in the FastAPI process, because mmcv has no wheel for the backend's Python
3.12 and this machine has no C++ compiler to build one. Keeping it in a
subprocess also means an MMAction2 crash cannot take the API down.

Contract:
    stdin  : JSON {"pose_results": [...], "img_shape": [h, w], "config": str,
                   "checkpoint": str, "label_map": str, "device": str, "topk": int}
    stdout : a single line  MMACTION_RESULT_JSON:{...}
    exit 0 on success, non-zero with a traceback on stderr otherwise.

Uses mmaction.apis.inference_skeleton so the config's own test_pipeline builds
the M x T x V x C tensor - no hand-rolled tensor shaping.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

RESULT_PREFIX = "MMACTION_RESULT_JSON:"


def main() -> int:
    payload = json.load(sys.stdin)

    config_path = payload["config"]
    checkpoint = payload["checkpoint"]
    label_map_path = payload["label_map"]
    device = payload.get("device", "cpu")
    topk = int(payload.get("topk", 5))
    img_shape = tuple(payload["img_shape"])  # (h, w)
    pose_results_raw = payload["pose_results"]

    import numpy as np
    import torch
    import mmengine
    from mmaction.apis import inference_skeleton, init_recognizer

    # Fall back to CPU rather than dying if cuda was requested but is absent.
    effective_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        effective_device = "cpu"

    # inference_skeleton expects numpy arrays per frame:
    #   keypoints       (num_person, V, 2)
    #   keypoint_scores (num_person, V)
    pose_results = [
        {
            "keypoints": np.array(f["keypoints"], dtype=np.float32),
            "keypoint_scores": np.array(f["keypoint_scores"], dtype=np.float32),
        }
        for f in pose_results_raw
    ]
    if not pose_results:
        raise ValueError("pose_results is empty - nothing to classify")

    num_keypoint = pose_results[0]["keypoints"].shape[1]

    cfg = mmengine.Config.fromfile(config_path)

    t0 = time.perf_counter()
    model = init_recognizer(cfg, checkpoint, device=effective_device)
    load_s = round(time.perf_counter() - t0, 2)

    # Cross-check the layout the checkpoint actually wants against what we send,
    # so a mismatch is reported loudly instead of producing garbage scores.
    graph_layout = None
    expected_nodes = None
    try:
        graph_cfg = cfg.model["backbone"].get("graph_cfg", {})
        graph_layout = graph_cfg.get("layout")
        expected_nodes = getattr(model.backbone.graph, "num_node", None)
    except (AttributeError, KeyError, TypeError):
        pass
    if expected_nodes is not None and num_keypoint != expected_nodes:
        raise ValueError(
            f"Keypoint count mismatch: the model graph (layout={graph_layout!r}) "
            f"expects {expected_nodes} nodes but received {num_keypoint}. "
            "Refusing to run, since the scores would be meaningless."
        )

    t1 = time.perf_counter()
    result = inference_skeleton(model, pose_results, img_shape)
    infer_s = round(time.perf_counter() - t1, 2)

    scores = result.pred_score.detach().cpu().numpy().astype(float)
    with open(label_map_path, encoding="utf-8") as fh:
        labels = [ln.strip() for ln in fh if ln.strip()]

    if len(labels) != len(scores):
        # Report rather than silently truncating - it would misname every class.
        note = (
            f"label map has {len(labels)} entries but the model produced "
            f"{len(scores)} scores"
        )
    else:
        note = None

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
        "top1": top[0] if top else None,
        "topk": top,
        "num_classes": int(len(scores)),
        "model_name": "ST-GCN",
        "config": config_path,
        "checkpoint": checkpoint,
        "graph_layout": graph_layout,
        "graph_num_node": expected_nodes,
        "num_keypoints_sent": int(num_keypoint),
        "num_frames_sent": len(pose_results),
        "num_persons_sent": int(pose_results[0]["keypoints"].shape[0]),
        "img_shape": [int(img_shape[0]), int(img_shape[1])],
        "device_requested": device,
        "device_used": effective_device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_load_s": load_s,
        "inference_s": infer_s,
        "label_map": label_map_path,
        "label_count": len(labels),
        "note": note,
    }
    print(RESULT_PREFIX + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - must surface the real traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
