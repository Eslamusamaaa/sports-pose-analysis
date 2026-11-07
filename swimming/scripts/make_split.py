"""Leakage-aware train/val split for the SwimXYZ pickle.

Rules:
  * all windows of one rendered sequence stay together (they overlap in time);
  * sequences that share an underlying motion (same pelvis-centred 3D pose
    fingerprint, i.e. the same motion rendered under another camera, light or
    water setting) stay together;
  * whole motion groups go to val, ~val_frac of each stroke's windows, chosen
    by a stable hash so the split is reproducible.

Why not hold out a swimmer body type: in SwimXYZ backstroke only ~102 distinct
motions exist, each rendered for every swimmer/camera/light (groups of up to
429 renders). Holding out a swimmer drags those whole groups into val and left
2 backstroke windows for training. Unseen MOTION is the leakage that matters -
a memorised motion rendered on another body would inflate accuracy.

Reports per-stroke / per-view / per-swimmer counts so skew is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="one or more converted pickles")
    ap.add_argument("--dst", default=r"D:\Users\karate-training\datasets\swimxyz\swimxyz_coco17_split.pkl")
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    anns = []
    for src in a.src:
        anns += pickle.load(open(src, "rb"))["annotations"]
    data = {"annotations": anns}

    # Group = connected component over (sequence) and (motion fingerprint).
    uf = UnionFind()
    for x in anns:
        seq = x["frame_dir"].split("#")[0]
        uf.union("seq:" + seq, "fp:" + x["swim_meta"]["fp"])

    # Windows per group, per stroke; then fill val per stroke in hash order.
    group_size: dict[str, Counter] = defaultdict(Counter)
    for x in anns:
        g = uf.find("seq:" + x["frame_dir"].split("#")[0])
        group_size[x["swim_meta"]["stroke"]][g] += 1
    val_groups: set[str] = set()
    for stroke, sizes in group_size.items():
        target = a.val_frac * sum(sizes.values())
        taken = 0
        for g in sorted(sizes, key=lambda g: hashlib.md5(g.encode()).hexdigest()):
            if taken >= target:
                break
            if taken + sizes[g] > 1.15 * target:
                continue  # a huge group would overshoot; try smaller ones
            val_groups.add(g)
            taken += sizes[g]
    split: dict[str, list[str]] = {"train": [], "val": []}
    count: dict[str, Counter] = defaultdict(Counter)
    for x in anns:
        g = uf.find("seq:" + x["frame_dir"].split("#")[0])
        part = "val" if g in val_groups else "train"
        split[part].append(x["frame_dir"])
        m = x["swim_meta"]
        count[part][m["stroke"]] += 1
        count[part]["view:" + m["view"]] += 1
        count[part]["swimmer:" + m["swimmer"]] += 1

    data["split"] = split
    with open(a.dst, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    groups = Counter(uf.find("seq:" + x["frame_dir"].split("#")[0]) for x in anns)
    report = {
        "val_frac": a.val_frac,
        "train_windows": len(split["train"]),
        "val_windows": len(split["val"]),
        "groups": len(groups),
        "largest_group_windows": max(groups.values()),
        "train": dict(sorted(count["train"].items())),
        "val": dict(sorted(count["val"].items())),
    }
    Path(a.dst + ".split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
