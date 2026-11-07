"""Training-time corruption that makes SwimXYZ ground truth look like OpenPose.

SwimXYZ joints are exact renders; OpenPose on real pool footage loses joints
under water and in splash, jitters, and sees swimmers at any scale and
direction. Without this the model learns from data that is too clean and
degrades on real video. Applied to RAW PIXEL keypoints, before PreNormalize2D.

Registered via ``custom_imports = dict(imports=['swim_transforms'])``; the
training command puts this folder on PYTHONPATH.
"""
from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS

# COCO-17 indices
LEFT = (1, 3, 5, 7, 9, 11, 13, 15)
RIGHT = (2, 4, 6, 8, 10, 12, 14, 16)
LIMBS = {
    "left_arm": (7, 9),
    "right_arm": (8, 10),
    "legs": (13, 14, 15, 16),  # below the waterline in most side views
    "head": (0, 1, 2, 3, 4),
}


@TRANSFORMS.register_module()
class SwimSkeletonAug(BaseTransform):
    def __init__(
        self,
        joint_drop: float = 0.08,
        limb_drop: float = 0.35,
        noise: float = 0.015,
        flip: float = 0.5,
        scale: tuple[float, float] = (0.5, 1.3),
    ) -> None:
        self.joint_drop = joint_drop
        self.limb_drop = limb_drop
        self.noise = noise
        self.flip = flip
        self.scale = scale

    def transform(self, results: dict) -> dict:
        kp = results["keypoint"].astype(np.float32).copy()  # (M, T, V, 2)
        sc = results["keypoint_score"].astype(np.float32).copy()  # (M, T, V)
        h, w = results["img_shape"]
        rng = np.random
        vis = sc > 0

        # Swim direction: mirror x and swap left/right joints.
        if rng.rand() < self.flip:
            kp[..., 0] = np.where(vis, w - 1 - kp[..., 0], 0)
            idx = np.arange(kp.shape[2])
            idx[list(LEFT)], idx[list(RIGHT)] = list(RIGHT), list(LEFT)
            kp, sc, vis = kp[:, :, idx], sc[:, :, idx], vis[:, :, idx]

        # Camera distance / position: scale about the visible centroid, then
        # shift, keeping the swimmer inside the frame when possible.
        if vis.any():
            c = kp[vis].mean(axis=0)
            s = rng.uniform(*self.scale)
            kp = np.where(vis[..., None], (kp - c) * s + c, 0)
            lo, hi = kp[vis].min(axis=0), kp[vis].max(axis=0)
            dx = rng.uniform(-lo[0], max(w - 1 - hi[0], -lo[0]))
            dy = rng.uniform(-lo[1], max(h - 1 - hi[1], -lo[1]))
            kp = np.where(vis[..., None], kp + np.array([dx, dy], np.float32), 0)

            # Jitter proportional to body size (pose-estimator noise).
            size = float(np.linalg.norm(hi - lo)) or 1.0
            kp = np.where(vis[..., None], kp + rng.normal(0, self.noise * size, kp.shape), 0)

        # Lose a whole limb for a contiguous stretch (under water / splash).
        T = kp.shape[1]
        if rng.rand() < self.limb_drop:
            joints = LIMBS[rng.choice(list(LIMBS))]
            t0 = rng.randint(0, T)
            t1 = min(T, t0 + rng.randint(T // 4, T + 1))
            vis[:, t0:t1, list(joints)] = False

        # Scattered single-joint misses.
        vis &= rng.rand(*vis.shape) >= self.joint_drop

        # Confidence: OpenPose-like, not a flat 1.0.
        sc = np.where(vis, rng.uniform(0.35, 0.95, sc.shape), 0).astype(np.float32)
        kp[~vis] = 0

        results["keypoint"] = kp.astype(np.float32)
        results["keypoint_score"] = sc
        return results
