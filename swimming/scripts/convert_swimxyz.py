"""SwimXYZ labels -> MMAction2 PoseDataset pickle (COCO-17, raw pixels).

Reads the *_labels.zip files directly (no extraction) and uses each sequence's
``body25/2D_cam.txt``: ground-truth BODY_25 joints projected into the 1920x1080
camera image. BODY_25 is exactly what OpenPose emits in the web app, so the
BODY_25 -> COCO-17 gather below is the same one the app applies at inference.

Choices that matter for matching inference:
  * 60 fps -> 30 fps (every 2nd frame). Real uploads are ~30 fps and the app's
    windows are counted in frames, so training at 60 fps would teach the model
    strokes that look twice as slow as they will at test time.
  * Contiguous 60-frame (2 s) windows, the same thing the app's sliding-window
    timeline sends to the model. UniformSampleFrames over a whole 5 s clip would
    instead time-compress the motion.
  * Joints outside the frame are zeroed with score 0, like OpenPose misses.

Grouping key for the split is written per sample (see make_split in this file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

W, H = 1920, 1080
SRC_FPS, DST_FPS = 60, 30
WINDOW, STRIDE = 60, 45  # frames at 30 fps -> 2.0 s windows, 3 per 5 s clip
MIN_VISIBLE = 0.6        # window kept if >= 60% of joint slots are in frame

STROKES = ("freestyle", "backstroke", "breaststroke", "butterfly")
ZIPS = {
    "freestyle": "Freestyle_labels.zip",
    "backstroke": "Backstroke_labels.zip",
    "breaststroke": "Breaststroke_labels.zip",
    "butterfly": "Butterfly_labels.zip",
}

# Same table as karate-ai-web/backend/app/services/keypoints.py (COCO_FROM_BODY25).
COCO_FROM_BODY25 = (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


def parse_txt(raw: bytes) -> np.ndarray:
    """'x;y;z;' rows with decimal commas -> (T, J, 3) float32."""
    lines = raw.decode("utf-8").splitlines()
    header = lines[0].rstrip(";").split(";")
    rows = [
        [float(v.replace(",", ".")) for v in ln.rstrip(";").split(";")]
        for ln in lines[1:]
        if ln.strip()
    ]
    arr = np.asarray(rows, dtype=np.float32)
    return arr.reshape(len(rows), len(header) // 3, 3)


def parse_meta(path: str) -> dict:
    # <Stroke>/<view>/Swimmer_Skin_a_Muscle_b/Water_Quantity_q_Height_h/
    #   Lighting_rotx_x_roty_y/Speed_s/position_p/body25/2D_cam.txt
    p = path.split("/")
    return {
        "view": p[1],
        "swimmer": p[2].replace("Swimmer_", ""),
        "water": p[3].replace("Water_", ""),
        "lighting": p[4].replace("Lighting_", ""),
        "speed": p[5],
        "position": p[6],
    }


def to_coco(b25: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = b25[:, COCO_FROM_BODY25, :2].copy()
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    score = inside.astype(np.float32)
    xy[~inside] = 0.0
    return xy, score


def convert(raw_dir: Path, out: Path, limit: int | None, strokes: list[str]) -> None:
    annotations: list[dict] = []
    stats: Counter = Counter()
    fingerprints: dict[str, list[str]] = defaultdict(list)

    for label, stroke in enumerate(STROKES):
        if stroke not in strokes:
            continue
        zpath = raw_dir / ZIPS[stroke]
        with zipfile.ZipFile(zpath) as z:
            infos = sorted(
                (i for i in z.infolist() if i.filename.endswith("body25/2D_cam.txt")),
                key=lambda i: i.filename,
            )
            if limit:
                infos = infos[:limit]
            for n, info in enumerate(infos):
                seq = info.filename.rsplit("/", 2)[0]
                meta = parse_meta(info.filename)
                b25 = parse_txt(z.read(info))[:: SRC_FPS // DST_FPS]
                xy, score = to_coco(b25)

                # Motion fingerprint from the pelvis-centred 3D pose, which does
                # not depend on camera, lighting or water - used to check whether
                # different renders share the same underlying motion.
                p3 = parse_txt(z.read(seq + "/body25/3D_pelvis.txt"))
                # Hash the WHOLE trajectory: many sequences share a start pose
                # (every backstroke begins alike), so a first-frame key collides.
                fp = hashlib.md5(np.round(p3[::5], 1).tobytes()).hexdigest()
                fingerprints[fp].append(f"{stroke}/{seq}")

                T = xy.shape[0]
                starts = list(range(0, max(T - WINDOW, 0) + 1, STRIDE))
                for k, s0 in enumerate(starts):
                    sc = score[s0:s0 + WINDOW]
                    if sc.shape[0] < WINDOW:
                        continue
                    if sc.mean() < MIN_VISIBLE:
                        stats["dropped_out_of_frame"] += 1
                        continue
                    annotations.append(
                        {
                            "frame_dir": f"{stroke}/{seq}#{k}",
                            "label": label,
                            "img_shape": (H, W),
                            "original_shape": (H, W),
                            "total_frames": WINDOW,
                            "keypoint": xy[None, s0:s0 + WINDOW],
                            "keypoint_score": sc[None],
                            "swim_meta": {**meta, "stroke": stroke, "fp": fp},
                        }
                    )
                    stats[f"{stroke}"] += 1
                    stats[f"view:{meta['view']}"] += 1
                if (n + 1) % 250 == 0:
                    print(f"  {stroke}: {n + 1}/{len(infos)} sequences", flush=True)
        print(f"{stroke}: done, {stats[stroke]} windows", flush=True)

    dup_groups = [v for v in fingerprints.values() if len(v) > 1]
    report = {
        "windows": len(annotations),
        "stats": dict(stats),
        "sequences": sum(len(v) for v in fingerprints.values()),
        "distinct_motions": len(fingerprints),
        "shared_motion_groups": len(dup_groups),
        "example_shared": dup_groups[:3],
        "window_frames": WINDOW,
        "stride_frames": STRIDE,
        "fps": DST_FPS,
        "img_shape": [H, W],
        "classes": list(STROKES),
        "strokes_converted": strokes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"annotations": annotations}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "example_shared"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/swimxyz/raw")
    ap.add_argument("--out", default="data/swimxyz/swimxyz_coco17_all.pkl")
    ap.add_argument("--limit", type=int, default=None, help="sequences per stroke (smoke test)")
    ap.add_argument("--strokes", default=",".join(STROKES), help="comma list; labels stay fixed")
    a = ap.parse_args()
    convert(Path(a.raw), Path(a.out), a.limit, a.strokes.split(","))
