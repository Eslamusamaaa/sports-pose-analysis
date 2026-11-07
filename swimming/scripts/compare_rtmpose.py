"""Swimming only: how often does RTMDet+RTMPose find the swimmer vs OpenPose?

Top-down: RTMDet finds person boxes, RTMPose (Halpe26 = COCO-17 + feet) runs on
the best box. A frame "has the swimmer" when >= 5 keypoints score > 0.3 - the
same >= 5 joints rule used for OpenPose in the web app.
Nothing here touches the web app or the karate pipeline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

LIBS = r"D:\Users\karate-training\swim\pose-libs"
sys.path.insert(0, LIBS)
from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.utils import adapt_mmdet_pipeline  # noqa: E402

M = Path(r"D:\Users\karate-training\swim\pose-models")
DET_CFG = Path(LIBS) / "mmpose/.mim/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_W = M / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CFG = Path(LIBS) / "mmpose/.mim/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
POSE_W = M / "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"


def main(videos: list[str], out_dir: str) -> None:
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    det = init_detector(str(DET_CFG), str(DET_W), device=dev)
    # mmpose's registry scope would otherwise hide mmdet's transforms.
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    pose = init_model(str(POSE_CFG), str(POSE_W), device=dev)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        cap = cv2.VideoCapture(v)
        found, n, t0 = [], 0, time.perf_counter()
        per_frame = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            r = inference_detector(det, frame).pred_instances
            keep = (r.labels == 0) & (r.scores > 0.3)
            boxes = r.bboxes[keep].cpu().numpy()
            scores = r.scores[keep].cpu().numpy()
            hit = False
            kps = None
            if len(boxes):
                best = boxes[int(np.argmax(scores))][None]
                p = inference_topdown(pose, frame, best)[0].pred_instances
                kp, ks = p.keypoints[0], p.keypoint_scores[0]
                hit = int((ks > 0.3).sum()) >= 5
                kps = np.concatenate([kp, ks[:, None]], 1).round(2).tolist()
            found.append(hit)
            per_frame.append(kps)
        dt = time.perf_counter() - t0
        name = Path(v).stem
        (out / f"{name}.rtmpose.json").write_text(json.dumps(per_frame))
        line = "".join("#" if h else "." for h in found)
        print(f"{name}: RTMPose found swimmer in {sum(found)}/{n} frames "
              f"({100 * sum(found) / max(n, 1):.0f}%), {dt / max(n, 1):.3f} s/frame", flush=True)
        for i in range(0, len(line), 40):
            print("   ", line[i:i + 40], flush=True)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
