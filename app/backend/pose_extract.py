"""RTMPose(+RTMDet) / YOLO11-pose -> OpenPose-format JSON + skeleton video.

Runs INSIDE the MMAction2 venv (torch + mmcv there), with the isolated
mmpose/mmdet and ultralytics folders put on sys.path by the caller's argv.
It writes one ``<frame>_keypoints.json`` per frame in OpenPose's own schema
(BODY_25 order, 75 floats per person), so the rest of the pipeline - tracking,
BODY_25 -> COCO-17, MMAction2, the skeleton video - runs unchanged for every
engine.

Mapping into BODY_25:
  * COCO-17 joints go to their exact BODY_25 slots.
  * Neck and MidHip do not exist in COCO: they are the midpoints of the
    shoulders / hips (dropped again by the BODY_25 -> COCO-17 converter).
  * Feet (big toe, small toe, heel): RTMPose-Halpe26 provides them; YOLO11
    (COCO-17) does not, so those slots stay 0 like an OpenPose miss.

Usage: python pose_extract.py <json-config>
  {"engine": "rtmpose"|"yolo11", "video": ..., "json_dir": ..., "out_video": ...,
   "libs": [...], "models": {...}, "device": "cuda:0", "max_frames": 0}
Progress lines on stdout are prefixed with @@POSE@@ (JSON).
"""
from __future__ import annotations

import json
import os
import sys
import time

# Ultralytics pip-installs missing packages at runtime by default (its tracker
# silently installed `lap` into the MMAction2 venv once). Never let it touch
# that venv: dependencies live in the isolated yolo-libs folder instead.
os.environ["YOLO_AUTOINSTALL"] = "False"

MSG = "@@POSE@@"

# BODY_25 slot <- COCO-17 index
B25_FROM_COCO = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 9: 12, 10: 14, 11: 16,
                 12: 11, 13: 13, 14: 15, 15: 2, 16: 1, 17: 4, 18: 3}
# BODY_25 slot <- Halpe26 index (feet)
B25_FROM_HALPE_FEET = {19: 20, 20: 22, 21: 24, 22: 21, 23: 23, 24: 25}
BONES = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
         (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
         (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
COLORS = [(255, 163, 77), (151, 220, 61), (77, 183, 255), (107, 107, 255)]


def emit(obj: dict) -> None:
    sys.stdout.write(MSG + json.dumps(obj) + "\n")
    sys.stdout.flush()


def to_body25(kp, sc, feet_kp=None, feet_sc=None) -> list[float]:
    """COCO-17 (+ optional Halpe feet) -> flat BODY_25 [x, y, c] * 25."""
    out = [[0.0, 0.0, 0.0] for _ in range(25)]
    for b, c in B25_FROM_COCO.items():
        out[b] = [float(kp[c][0]), float(kp[c][1]), float(sc[c])]
    for b, (l, r) in ((1, (5, 6)), (8, (11, 12))):  # Neck, MidHip
        if sc[l] > 0 and sc[r] > 0:
            out[b] = [float(kp[l][0] + kp[r][0]) / 2, float(kp[l][1] + kp[r][1]) / 2,
                      float(min(sc[l], sc[r]))]
    if feet_kp is not None:
        for b, h in B25_FROM_HALPE_FEET.items():
            out[b] = [float(feet_kp[h][0]), float(feet_kp[h][1]), float(feet_sc[h])]
    return [v for j in out for v in j]


def appearance(frame, bbox) -> list[float] | None:
    """Clothing colour signature of one person: HSV histogram of the upper half
    (shirt) and the lower half (trousers / shin guards) of the box.

    Used to tell the two fighters apart from each other and from the coach
    even when the tracker swaps or loses ids. H8 x S2 x V3 per half - V is
    needed because black vs white clothing differs only in brightness.
    """
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    # Inset the box a little so the background at the edges counts less.
    dx, dy = (x2 - x1) // 6, (y2 - y1) // 20
    crop = frame[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mid = hsv.shape[0] // 2
    out: list[float] = []
    for part in (hsv[:mid], hsv[mid:]):
        hist = cv2.calcHist([part], [0, 1, 2], None, [8, 2, 3], [0, 180, 0, 256, 0, 256]).flatten()
        total = float(hist.sum()) or 1.0
        out += [round(float(v) / total, 3) for v in hist]
    return out


def build_reid(cfg: dict):
    """OSNet-AIN x1.0 (MSMT17, MIT licence) person re-identification.

    Returns embed(frame, boxes) -> list of 512-d L2-normalised float32 vectors
    (None for a box too small to use), or None when re-id is not configured.
    A re-id network is trained to answer "is this the SAME person?" from the
    whole appearance (clothes, build, hair, shoes); far more reliable than a
    colour histogram when two fighters wear similar colours or turn around.
    """
    rc = cfg.get("reid")
    if not rc:
        return None
    import numpy as np
    import torch

    if rc["lib"] not in sys.path:
        sys.path.insert(0, rc["lib"])
    import osnet_ain  # vendored model definition from Torchreid (MIT)

    ck = torch.load(rc["weights"], map_location="cpu")
    sd = {k.replace("module.", "", 1): v for k, v in ck.get("state_dict", ck).items()}
    model = osnet_ain.osnet_ain_x1_0(num_classes=sd["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(sd, strict=False)
    dev = cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def embed(frame, boxes):
        import cv2

        h, w = frame.shape[:2]
        crops, keep = [], []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (128, 256)).astype(np.float32) / 255.0  # the size it was trained on
            crops.append(((crop - mean) / std).transpose(2, 0, 1))
            keep.append(i)
        out = [None] * len(boxes)
        if not crops:
            return out
        with torch.no_grad():
            feats = model(torch.from_numpy(np.stack(crops)).to(dev))
            feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        for i, f in zip(keep, feats):
            out[i] = f
        return out

    return embed


def person_json(p: dict) -> dict:
    """One person in OpenPose's JSON schema, plus the detector's box/track.

    ``person_id`` is where OpenPose would put an identity (it always writes -1);
    YOLO11's ByteTrack id goes there so the pipeline can use real tracking.
    ``bbox`` / ``bbox_score`` are extra keys that OpenPose parsers ignore.
    """
    out = {"person_id": [p["id"] if p.get("id") is not None else -1],
           "pose_keypoints_2d": p["kp"]}
    if p.get("bbox") is not None:
        out["bbox"] = [round(float(v), 1) for v in p["bbox"]]
        out["bbox_score"] = round(float(p.get("score", 0.0)), 3)
    if p.get("app") is not None:
        out["appearance"] = p["app"]
    if p.get("reid_idx") is not None:
        out["reid_idx"] = p["reid_idx"]  # row in reid.npy (batch)
    if p.get("reid_b64") is not None:
        out["reid_b64"] = p["reid_b64"]  # inline float16 (live)
    return out


def build_infer(cfg: dict):
    """Load the chosen engine; return frame -> list of people.

    Each person: {"kp": flat BODY_25 (75 floats), "bbox": [x1, y1, x2, y2],
    "score": detector confidence, "id": tracker id or None}. RTMDet supplies
    the boxes for RTMPose (no tracker ids); YOLO11 supplies boxes, keypoints
    and ByteTrack ids in one pass. Shared by batch (this file) and live
    (pose_live.py).
    """
    for p in cfg.get("libs", []):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np

    engine, models, dev = cfg["engine"], cfg["models"], cfg.get("device", "cuda:0")
    if engine == "rtmpose":
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import inference_topdown, init_model
        from mmpose.utils import adapt_mmdet_pipeline

        det = init_detector(models["det_config"], models["det_weights"], device=dev)
        det.cfg = adapt_mmdet_pipeline(det.cfg)
        pose = init_model(models["pose_config"], models["pose_weights"], device=dev)

        def infer(frame):
            r = inference_detector(det, frame).pred_instances
            keep = (r.labels == 0) & (r.scores > 0.3)
            boxes = r.bboxes[keep].cpu().numpy()
            scores = r.scores[keep].cpu().numpy()
            order = np.argsort(-scores)[: cfg.get("max_people", 8)]
            boxes, scores = boxes[order], scores[order]
            if not len(boxes):
                return []
            people = []
            for p, box, sc in zip(inference_topdown(pose, frame, boxes), boxes, scores):
                k, s = p.pred_instances.keypoints[0], p.pred_instances.keypoint_scores[0]
                s = np.clip(s, 0.0, 1.0)  # SimCC scores can exceed 1
                people.append({"kp": to_body25(k[:17], s[:17], k, s), "bbox": box.tolist(),
                               "score": float(sc), "id": None})
            return people
    elif engine == "yolo11":
        from ultralytics import YOLO

        model = YOLO(models["yolo_weights"])
        ydev = 0 if dev.startswith("cuda") else "cpu"

        def infer(frame):
            # track() = detect + pose + ByteTrack in one call; persist=True keeps
            # identities across successive frames of this video / camera.
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                            device=ydev, verbose=False)[0]
            if r.keypoints is None or r.boxes is None or not len(r.boxes):
                return []
            k = r.keypoints.xy.cpu().numpy()
            s = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(k.shape[:2])
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(k)
            n = min(len(k), cfg.get("max_people", 8))
            return [{"kp": to_body25(k[i], s[i]), "bbox": boxes[i].tolist(),
                     "score": float(confs[i]), "id": ids[i]} for i in range(n)]
    else:
        raise SystemExit(f"unknown engine {engine}")
    return infer


def draw(frame, people) -> None:
    """Skeleton + bounding box + label (track id and detector confidence)."""
    import cv2

    for pi, person in enumerate(people):
        flat = person["kp"]
        # Colour by track id when there is one, so a fighter keeps one colour.
        key = person["id"] if person.get("id") is not None else pi
        c = COLORS[key % len(COLORS)]
        if person.get("bbox") is not None:
            x1, y1, x2, y2 = (int(v) for v in person["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            label = (f"id {person['id']} " if person.get("id") is not None else "") + f"{person.get('score', 0):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), c, -1)
            cv2.putText(frame, label, (x1 + 3, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        pts = [(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]) for i in range(25)]
        for a, b in BONES:
            if pts[a][2] > 0.1 and pts[b][2] > 0.1:
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), c, 2)
        for x, y, s in pts:
            if s > 0.1:
                cv2.circle(frame, (int(x), int(y)), 3, c, -1)


def box_from_keypoints(flat, w: int, h: int):
    """Person box from the joints (OpenPose gives no box): joint extent padded
    by a share of its height, since joints stop short of the head top / soles."""
    xs = [flat[i * 3] for i in range(25) if flat[i * 3 + 2] > 0.1]
    ys = [flat[i * 3 + 1] for i in range(25) if flat[i * 3 + 2] > 0.1]
    if len(xs) < 4:
        return None
    bh = max(ys) - min(ys)
    pad_x, pad_y = 0.15 * bh, 0.12 * bh
    return [max(min(xs) - pad_x, 0), max(min(ys) - pad_y, 0),
            min(max(xs) + pad_x, w), min(max(ys) + pad_y, h)]


def reid_pass(cfg: dict) -> int:
    """Add re-id embeddings to an EXISTING JSON folder (engines without them,
    e.g. OpenPose): who-is-who then works the same whatever pose model ran.
    Writes reid.npy next to the folder and a reid_idx on each person."""
    from pathlib import Path

    import cv2
    import numpy as np

    embed = build_reid(cfg)
    if embed is None:
        emit({"error": "re-id model not configured"})
        return 1
    json_dir = Path(cfg["json_dir"])
    files = sorted(json_dir.glob("*_keypoints.json"))
    cap = cv2.VideoCapture(cfg["video"])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    emit({"ready": True, "engine": "reid", "frames": len(files)})
    rows: list = []
    t0 = time.perf_counter()
    for n, path in enumerate(files):
        ok, frame = cap.read()
        if not ok:
            break
        data = json.loads(path.read_text())
        people = data.get("people") or []
        boxes, idx = [], []
        for i, p in enumerate(people):
            b = p.get("bbox") or box_from_keypoints(p.get("pose_keypoints_2d") or [], w, h)
            if b is not None:
                boxes.append(b)
                idx.append(i)
        if boxes:
            for i, f in zip(idx, embed(frame, boxes)):
                if f is not None:
                    people[i]["reid_idx"] = len(rows)
                    rows.append(f.astype("float16"))
            path.write_text(json.dumps(data))
        if n % 100 == 0:
            emit({"progress": n, "total": len(files)})
    cap.release()
    if rows:
        np.save(json_dir.parent / "reid.npy", np.stack(rows))
    emit({"done": True, "frames": len(files), "embeddings": len(rows),
          "duration_s": round(time.perf_counter() - t0, 2)})
    return 0


def track_pass(cfg: dict) -> int:
    """Give tracker ids to an EXISTING JSON folder whose engine has none
    (RTMPose, OpenPose): the same ByteTrack YOLO11 runs inside model.track(),
    fed with the engine's boxes (or a box around the joints). Writes the id
    into person_id, exactly where YOLO11's ids go."""
    for p in cfg.get("libs", []):
        if p not in sys.path:
            sys.path.insert(0, p)
    from pathlib import Path

    import numpy as np
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    class Dets:  # the minimal "Boxes" interface BYTETracker reads
        def __init__(self, xyxy, conf):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = np.zeros(len(conf))
            self.xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]]) \
                if len(conf) else np.zeros((0, 4))

        def __len__(self):
            return len(self.conf)

        def __getitem__(self, idx):
            return Dets(self.xyxy[idx], self.conf[idx])

    args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(args)
    json_dir = Path(cfg["json_dir"])
    files = sorted(json_dir.glob("*_keypoints.json"))
    w, h = cfg.get("width") or 1e9, cfg.get("height") or 1e9
    emit({"ready": True, "engine": "track", "frames": len(files)})
    n_ids = set()
    for n, path in enumerate(files):
        data = json.loads(path.read_text())
        people = data.get("people") or []
        boxes, confs, idx = [], [], []
        for i, p in enumerate(people):
            flat = p.get("pose_keypoints_2d") or []
            b = p.get("bbox") or box_from_keypoints(flat, w, h)
            if b is None:
                continue
            c = p.get("bbox_score")
            if c is None:  # OpenPose: mean confidence of the detected joints
                cs = [flat[i * 3 + 2] for i in range(len(flat) // 3) if flat[i * 3 + 2] > 0.1]
                c = float(sum(cs) / len(cs)) if cs else 0.0
            boxes.append(b)
            confs.append(c)
            idx.append(i)
        dets = Dets(np.asarray(boxes, float).reshape(-1, 4), np.asarray(confs, float))
        out = tracker.update(dets)
        for p in people:
            p["person_id"] = [-1]
        for row in out:
            tid, di = int(row[4]), int(row[-1])
            people[idx[di]]["person_id"] = [tid]
            n_ids.add(tid)
        path.write_text(json.dumps(data))
        if n % 200 == 0:
            emit({"progress": n, "total": len(files)})
    emit({"done": True, "frames": len(files), "tracks": len(n_ids)})
    return 0


def main() -> int:
    cfg = json.loads(sys.argv[1])
    if cfg.get("mode") == "reid":
        return reid_pass(cfg)
    if cfg.get("mode") == "track":
        return track_pass(cfg)
    from pathlib import Path

    import cv2

    engine = cfg["engine"]
    t_load = time.perf_counter()
    infer = build_infer(cfg)
    embed = build_reid(cfg)
    load_s = time.perf_counter() - t_load
    reid_rows: list = []  # one 512-d float16 row per detected person, in order

    json_dir = Path(cfg["json_dir"])
    json_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cfg["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = cfg.get("max_frames") or 0
    writer = None
    if cfg.get("out_video"):
        writer = cv2.VideoWriter(cfg["out_video"], cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    emit({"ready": True, "engine": engine, "load_s": round(load_s, 2), "frames": total})

    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (limit and n >= limit):
            break
        people = infer(frame)
        for person in people:  # colour signature from the CLEAN frame (before drawing)
            if person.get("bbox") is not None:
                person["app"] = appearance(frame, person["bbox"])
        if embed is not None and people:
            with_box = [p for p in people if p.get("bbox") is not None]
            for p, f in zip(with_box, embed(frame, [p["bbox"] for p in with_box])):
                if f is not None:
                    p["reid_idx"] = len(reid_rows)
                    reid_rows.append(f.astype("float16"))
        (json_dir / f"{n:012d}_keypoints.json").write_text(json.dumps({
            "version": 1.3,
            "people": [person_json(p) for p in people],
        }))
        if writer is not None:
            draw(frame, people)
            writer.write(frame)
        n += 1
        if n % 50 == 0:
            emit({"progress": n, "total": total})
    cap.release()
    if writer is not None:
        writer.release()
    if reid_rows:
        import numpy as np

        # One binary file next to the JSON folder: 512 numbers per person per
        # frame would make every JSON file ~3 KB per person.
        np.save(json_dir.parent / "reid.npy", np.stack(reid_rows))
    dt = time.perf_counter() - t0
    emit({"done": True, "frames": n, "duration_s": round(dt, 2),
          "seconds_per_frame": round(dt / max(n, 1), 4), "load_s": round(load_s, 2),
          "width": w, "height": h})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc(file=sys.stderr)
        emit({"error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
