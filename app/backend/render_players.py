"""Draw everyone, and name the players: grey box + skeleton for each person,
colour + "Player 1"/"Player 2" for the two players.

Players are decided after tracking, over the whole clip (the most-moving,
best-tracked people - the same ones sent to the action model), so this runs
as a separate pass once they are known. Everyone else (coach, referee, spectators)
is drawn in thin grey with no label.
Works for every pose engine: boxes come from the detector when it has them
(RTMDet / YOLO11) and from the joint extent otherwise (OpenPose).

Runs in the MMAction2 venv (has OpenCV). Usage:
  python render_players.py <video> <players.json> <out.avi>
players.json: {"labels": ["Player 1", ...],
               "frames": [[{"slot": 0, "kp": [75 floats], "bbox": [x1,y1,x2,y2] | null}, ...], ...]}
"""
from __future__ import annotations

import json
import sys

import cv2

# BGR, one fixed colour per player slot (Player 1 blue, Player 2 orange).
SLOT_COLORS = [(255, 160, 40), (40, 150, 255), (90, 220, 90), (220, 90, 220)]
OTHER_COLOR = (170, 170, 170)  # everyone who is not a player
BONES = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
         (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
         (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
MIN_C = 0.1


def box_from_joints(pts, w, h):
    xs = [x for x, y, c in pts if c > MIN_C]
    ys = [y for x, y, c in pts if c > MIN_C]
    if len(xs) < 2:
        return None
    pad_x, pad_y = 0.12 * (max(xs) - min(xs) + 10), 0.08 * (max(ys) - min(ys) + 10)
    return [max(min(xs) - pad_x, 0), max(min(ys) - pad_y, 0),
            min(max(xs) + pad_x, w - 1), min(max(ys) + pad_y, h - 1)]


def main() -> int:
    video, players_path, out_path = sys.argv[1:4]
    data = json.load(open(players_path, encoding="utf-8"))
    labels, frames = data["labels"], data["frames"]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    thick = max(2, round(min(w, h) / 360))
    font = max(0.5, min(w, h) / 900)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = frames[n] if n < len(frames) else []
        # Non-players first, players on top so their boxes are never hidden.
        for p in sorted(row, key=lambda q: q["slot"] >= 0):
            is_player = p["slot"] >= 0
            c = SLOT_COLORS[p["slot"] % len(SLOT_COLORS)] if is_player else OTHER_COLOR
            t = thick if is_player else max(1, thick // 2)
            pts = [(p["kp"][i * 3], p["kp"][i * 3 + 1], p["kp"][i * 3 + 2]) for i in range(25)]
            for a, b in BONES:
                if pts[a][2] > MIN_C and pts[b][2] > MIN_C:
                    cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])),
                             c, t, cv2.LINE_AA)
            for x, y, s in pts:
                if s > MIN_C:
                    cv2.circle(frame, (int(x), int(y)), t + 1, c, -1, cv2.LINE_AA)
            box = p.get("bbox") or box_from_joints(pts, w, h)
            if not box:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, t)
            if not is_player:
                continue  # everyone is detected, only the two players are named
            label = labels[p["slot"]]
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
            ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 8, ty + base), c, -1)
            cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (255, 255, 255), thick, cv2.LINE_AA)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    print(json.dumps({"frames": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
