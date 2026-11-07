"""Honest evaluation of the swimming ST-GCN on the held-out split.

Runs the val split twice through the trained model:
  clean     - exact SwimXYZ joints (upper bound, what training validated on)
  corrupted - same SwimSkeletonAug used in training (dropped limbs, jitter),
              a rough stand-in for OpenPose on real pool footage

and reports accuracy per camera view, per stroke, and a confusion matrix, so a
weak view (e.g. underwater) is visible instead of hidden in one average.

Real-footage accuracy is still UNKNOWN after this - it needs labelled real video.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope

STROKES = ["freestyle", "backstroke", "breaststroke", "butterfly"]


def run(model, samples, pipeline, batch=128):
    # test_step applies the model's own data_preprocessor and clip averaging,
    # exactly like the training-time validation loop.
    from mmengine.dataset import pseudo_collate

    preds = []
    for i in range(0, len(samples), batch):
        chunk = pseudo_collate([pipeline(dict(s)) for s in samples[i:i + batch]])
        with torch.no_grad():
            out = model.test_step(chunk)
        preds.extend(int(o.pred_score.argmax()) for o in out)
    return np.array(preds)


def report(name, y, p, views):
    out = {"overall_acc": round(float((y == p).mean()), 4)}
    out["per_view"] = {v: round(float((y[views == v] == p[views == v]).mean()), 4)
                       for v in sorted(set(views))}
    out["per_stroke_recall"] = {s: round(float((p[y == i] == i).mean()), 4)
                                for i, s in enumerate(STROKES)}
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    out["confusion_rows_true_cols_pred"] = cm.tolist()
    print(f"\n== {name}: overall {out['overall_acc']:.4f}")
    for v, a in out["per_view"].items():
        print(f"   {v:18s} {a:.4f}")
    print("   confusion (rows=true, cols=pred):", STROKES)
    for s, row in zip(STROKES, cm):
        print(f"   {s:13s}", row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"D:\Users\karate-training\swim\stgcn_swim4.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=r"D:\Users\karate-training\datasets\swimxyz\swimxyz_coco17_split.pkl")
    ap.add_argument("--out", default=r"D:\Users\karate-training\swim\eval_report.json")
    a = ap.parse_args()

    import swim_transforms  # noqa: F401  registers SwimSkeletonAug
    from mmaction.apis import init_recognizer

    cfg = Config.fromfile(a.config)
    init_default_scope("mmaction")
    model = init_recognizer(cfg, a.checkpoint, device="cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    data = pickle.load(open(a.data, "rb"))
    val_ids = set(data["split"]["val"])
    samples = [x for x in data["annotations"] if x["frame_dir"] in val_ids]
    y = np.array([x["label"] for x in samples])
    views = np.array([x["swim_meta"]["view"] for x in samples])

    clean = Compose(cfg.val_pipeline)
    np.random.seed(0)
    corrupt = Compose([dict(type="SwimSkeletonAug")] + list(cfg.val_pipeline))

    res = {
        "val_windows": len(samples),
        "checkpoint": a.checkpoint,
        "clean": report("clean", y, run(model, samples, clean), views),
        "corrupted": report("corrupted (OpenPose-like)", y, run(model, samples, corrupt), views),
        "note": "Synthetic held-out swimmer only. Accuracy on real pool video is not measured here.",
    }
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()
