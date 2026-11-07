"""Keypoint layout definitions for OpenPose BODY_25 and COCO-17.

These orderings are not guesses. BODY_25 is taken from OpenPose's own
`include/openpose/pose/poseParameters.hpp` / `POSE_BODY_25_BODY_PARTS`, and
COCO-17 is the standard COCO `person_keypoints` ordering, which is what
MMAction2's `graph_cfg=dict(layout='coco')` expects (see
mmaction/models/utils/graph.py -> `coco` layout, num_node=17).
"""
from __future__ import annotations

# OpenPose BODY_25 output order (25 joints).
BODY_25_NAMES: tuple[str, ...] = (
    "Nose",        # 0
    "Neck",        # 1
    "RShoulder",   # 2
    "RElbow",      # 3
    "RWrist",      # 4
    "LShoulder",   # 5
    "LElbow",      # 6
    "LWrist",      # 7
    "MidHip",      # 8
    "RHip",        # 9
    "RKnee",       # 10
    "RAnkle",      # 11
    "LHip",        # 12
    "LKnee",       # 13
    "LAnkle",      # 14
    "REye",        # 15
    "LEye",        # 16
    "REar",        # 17
    "LEar",        # 18
    "LBigToe",     # 19
    "LSmallToe",   # 20
    "LHeel",       # 21
    "RBigToe",     # 22
    "RSmallToe",   # 23
    "RHeel",       # 24
)

# Standard COCO-17 order.
COCO_17_NAMES: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

# BODY_25 index that supplies each COCO-17 index.
# COCO_FROM_BODY25[coco_idx] = body25_idx
#
# Every COCO joint has an exact BODY_25 counterpart, so this is a pure
# gather/subset - no interpolation, no synthesised joints.
COCO_FROM_BODY25: tuple[int, ...] = (
    0,   # nose           <- Nose
    16,  # left_eye       <- LEye
    15,  # right_eye      <- REye
    18,  # left_ear       <- LEar
    17,  # right_ear      <- REar
    5,   # left_shoulder  <- LShoulder
    2,   # right_shoulder <- RShoulder
    6,   # left_elbow     <- LElbow
    3,   # right_elbow    <- RElbow
    7,   # left_wrist     <- LWrist
    4,   # right_wrist    <- RWrist
    12,  # left_hip       <- LHip
    9,   # right_hip      <- RHip
    13,  # left_knee      <- LKnee
    10,  # right_knee     <- RKnee
    14,  # left_ankle     <- LAnkle
    11,  # right_ankle    <- RAnkle
)

# BODY_25 joints with no COCO-17 equivalent. They are dropped by the
# conversion; documented so the loss is explicit rather than silent.
BODY25_DROPPED: tuple[int, ...] = (1, 8, 19, 20, 21, 22, 23, 24)

assert len(BODY_25_NAMES) == 25
assert len(COCO_17_NAMES) == 17
assert len(COCO_FROM_BODY25) == 17
assert len(set(COCO_FROM_BODY25)) == 17
assert set(COCO_FROM_BODY25) | set(BODY25_DROPPED) == set(range(25))


def conversion_table() -> list[dict]:
    """Human-readable mapping, surfaced through the API for documentation."""
    return [
        {
            "coco_index": ci,
            "coco_name": COCO_17_NAMES[ci],
            "body25_index": bi,
            "body25_name": BODY_25_NAMES[bi],
        }
        for ci, bi in enumerate(COCO_FROM_BODY25)
    ]


def dropped_table() -> list[dict]:
    return [{"body25_index": i, "body25_name": BODY_25_NAMES[i]} for i in BODY25_DROPPED]
