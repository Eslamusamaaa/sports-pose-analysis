"""BODY_25 (OpenPose) -> COCO-17 conversion for MMAction2 skeleton models.

WHY A CONVERSION IS REQUIRED
----------------------------
The selected checkpoint is ST-GCN trained on NTU RGB+D 60 with the 2D-keypoint
config::

    configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
        backbone = dict(type='STGCN', graph_cfg=dict(layout='coco', ...))
        cls_head = dict(type='GCNHead', num_classes=60, in_channels=256)

``layout='coco'`` resolves in ``mmaction/models/utils/graph.py`` to
``num_node = 17`` with a fixed COCO adjacency. OpenPose BODY_25 emits 25 joints
in a different order. Feeding 25 joints to a 17-node graph is a hard shape
mismatch, so BODY_25 is NOT directly compatible and a mapping is mandatory.

THE MAPPING
-----------
Every COCO-17 joint has an exact BODY_25 counterpart, so this is a pure
gather (subset + reorder). No joint is invented, interpolated or averaged.
See ``keypoints.COCO_FROM_BODY25`` for the full table.

Eight BODY_25 joints have no COCO-17 equivalent and are DISCARDED:
    1  Neck        8  MidHip
    19 LBigToe     20 LSmallToe   21 LHeel
    22 RBigToe     23 RSmallToe   24 RHeel

TENSOR CONTRACT
---------------
``mmaction.apis.inference_skeleton(model, pose_results, img_shape)`` expects
``pose_results`` as one dict per frame::

    {'keypoints':       np.ndarray (num_person, 17, 2)   # raw pixel coords
     'keypoint_scores': np.ndarray (num_person, 17)}

and builds the M x T x V x C tensor itself, then runs the config's own
test_pipeline (PreNormalize2D -> GenSkeFeat -> UniformSampleFrames(clip_len=100,
num_clips=10) -> PoseDecode -> FormatGCNInput(num_person=2) -> PackActionInputs).
So:
    M = 2 persons (zero-padded by FormatGCNInput)
    T = 100 frames per clip, 10 clips averaged
    V = 17 keypoints
    C = 3 (x, y, score) - STGCN in_channels defaults to 3

Coordinates must stay in RAW PIXELS here; PreNormalize2D does the
``(x - w/2) / (w/2)`` normalisation using img_shape. Normalising twice would
silently wreck the input, so this module deliberately does not scale anything.

KNOWN LIMITATIONS (do not paper over these)
-------------------------------------------
* The NTU RGB+D 2D skeletons this checkpoint was trained on were produced by a
  top-down HRNet pose estimator, not by OpenPose. Joint localisation and
  confidence distributions differ, so there is a domain shift even after the
  layout is made compatible.
* Dropping Neck/MidHip removes the torso-centre cues BODY_25 provides; COCO's
  graph centre is the nose instead.
* NTU RGB+D 60 contains 60 everyday actions recorded indoors from 3 fixed
  cameras. It contains NO karate classes, so any karate clip is out-of-domain
  and its prediction is not a karate label in any meaningful sense.
"""
from __future__ import annotations

from ..utils.logging_utils import JobLogger
from .keypoints import (
    BODY25_DROPPED,
    BODY_25_NAMES,
    COCO_17_NAMES,
    COCO_FROM_BODY25,
    conversion_table,
    dropped_table,
)


class ConversionError(RuntimeError):
    pass


def convert_person_body25_to_coco17(
    joints: list[tuple[float, float, float]],
) -> tuple[list[list[float]], list[float]]:
    """Gather one person's 25 joints into COCO-17 order.

    Returns (keypoints_xy[17][2], scores[17]).
    """
    if len(joints) != 25:
        raise ConversionError(f"expected 25 BODY_25 joints, got {len(joints)}")
    xy: list[list[float]] = []
    scores: list[float] = []
    for body25_idx in COCO_FROM_BODY25:
        x, y, c = joints[body25_idx]
        xy.append([x, y])
        scores.append(c)
    return xy, scores


def convert(
    parsed: dict,
    log: JobLogger,
    max_persons: int = 2,
    min_track_frames: int = 1,
) -> dict:
    """Build per-frame COCO-17 pose_results from the tracked BODY_25 skeleton.

    Person slots are filled from the highest-scoring tracks so that slot k refers
    to the same human across all frames. Frames where a track is absent are
    zero-filled, which PreNormalize2D/PoseDecode treat as missing.
    """
    frames = parsed["frames"]
    if not frames:
        raise ConversionError("no frames to convert")

    tracks = [t for t in parsed.get("tracks", []) if t["frames_present"] >= min_track_frames]
    if not tracks:
        raise ConversionError(
            "No usable person tracks were found. OpenPose detected no people "
            "with sufficient confidence, so there is nothing to classify."
        )

    chosen = tracks[:max_persons]
    chosen_ids = [t["track_id"] for t in chosen]
    log.info(
        f"conversion: BODY_25(25) -> COCO-17(17); using {len(chosen_ids)} "
        f"person slot(s) from track ids {chosen_ids}",
        stage="convert",
    )

    pose_results: list[dict] = []
    filled = 0
    empty_slots = 0
    interpolated = 0
    for frame in frames:
        by_track = {p.get("track_id"): p for p in frame["people"] if p.get("track_id") is not None}
        kp_frame: list[list[list[float]]] = []
        sc_frame: list[list[float]] = []
        for tid in chosen_ids:
            person = by_track.get(tid)
            if person is None:
                kp_frame.append([[0.0, 0.0] for _ in range(17)])
                sc_frame.append([0.0] * 17)
                empty_slots += 1
            else:
                xy, sc = convert_person_body25_to_coco17(person["keypoints"])
                interpolated += bool(person.get("interpolated"))
                kp_frame.append(xy)
                sc_frame.append(sc)
                filled += 1
        pose_results.append({"keypoints": kp_frame, "keypoint_scores": sc_frame})

    total_slots = len(frames) * len(chosen_ids)
    log.info(
        f"conversion completed: {len(pose_results)} frames x {len(chosen_ids)} persons "
        f"x 17 keypoints ({filled}/{total_slots} slots filled, "
        f"{empty_slots} zero-padded, {interpolated} of the filled ones interpolated)",
        stage="convert",
    )

    return {
        "pose_results": pose_results,
        "meta": {
            "source_format": "BODY_25",
            "source_num_keypoints": 25,
            "target_format": "COCO-17",
            "target_num_keypoints": 17,
            "target_layout": "coco",
            "num_frames": len(pose_results),
            "num_persons": len(chosen_ids),
            "track_ids_used": chosen_ids,
            "slots_filled": filled,
            "slots_zero_padded": empty_slots,
            "slots_interpolated": interpolated,
            "coordinate_space": "raw pixels (PreNormalize2D normalises downstream)",
            "tensor_shape_note": (
                "inference_skeleton builds M x T x V x C = "
                f"2 x 100 x 17 x 3 per clip (10 clips averaged) from these "
                f"{len(pose_results)} frames"
            ),
            "mapping": conversion_table(),
            "dropped_body25_joints": dropped_table(),
            "body25_names": list(BODY_25_NAMES),
            "coco17_names": list(COCO_17_NAMES),
            "dropped_count": len(BODY25_DROPPED),
        },
    }
