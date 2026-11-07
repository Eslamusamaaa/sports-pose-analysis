"""Live webcam pose with RTMPose or YOLO11, written exactly like OpenPose live.

OpenPoseDemo.exe --camera writes, per frame, ``<n>_keypoints.json`` and
``<n>_rendered.jpg`` into two folders that realtime_service's watcher reads.
This process does the same with another engine, so the live session (tracking,
classification, "not swimming" gate, Stop) is identical for every engine.

The JPEG is written before the JSON, and both via a temp file + rename, so the
watcher never sees a half-written file or a JSON whose image is missing.
Runs in the MMAction2 venv; config arrives as one JSON argv like pose_extract.
"""
from __future__ import annotations

import json
import os
import sys
import time

from pose_extract import appearance, build_infer, build_reid, emit, person_json


def main() -> int:
    cfg = json.loads(sys.argv[1])
    import cv2

    infer = build_infer(cfg)
    embed = build_reid(cfg)
    cam = cv2.VideoCapture(int(cfg.get("camera_index", 0)), cv2.CAP_DSHOW)
    w, h = (int(v) for v in str(cfg.get("resolution", "640x480")).split("x"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, no backlog
    if not cam.isOpened():
        emit({"error": f"cannot open camera {cfg.get('camera_index', 0)}"})
        return 2
    json_dir, img_dir = cfg["json_dir"], cfg["img_dir"]
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    emit({"ready": True, "engine": cfg["engine"]})

    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue
        people = infer(frame)
        for person in people:  # colour signature from the CLEAN frame (before drawing)
            if person.get("bbox") is not None:
                person["app"] = appearance(frame, person["bbox"])
        if embed is not None and people:
            import base64

            with_box = [p for p in people if p.get("bbox") is not None]
            for p, f in zip(with_box, embed(frame, [p["bbox"] for p in with_box])):
                if f is not None:
                    p["reid_b64"] = base64.b64encode(f.astype("float16").tobytes()).decode("ascii")
        stem = f"{n:012d}"
        img_tmp = os.path.join(img_dir, stem + ".jpg.part")  # not *.jpg: the watcher prunes *.jpg
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with open(img_tmp, "wb") as fh:
            fh.write(buf.tobytes())
        os.replace(img_tmp, os.path.join(img_dir, stem + "_rendered.jpg"))
        js_tmp = os.path.join(json_dir, stem + ".tmp")
        with open(js_tmp, "w") as fh:
            json.dump({"version": 1.3,
                       "people": [person_json(p) for p in people]}, fh)
        os.replace(js_tmp, os.path.join(json_dir, stem + "_keypoints.json"))
        n += 1
        if n % 100 == 0:
            emit({"frames": n, "fps": round(n / (time.perf_counter() - t0), 1)})


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc(file=sys.stderr)
        emit({"error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
