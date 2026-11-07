"""Step 2 (venv with OpenCV): one 2x4 image per window (2 columns, 4 rows).

Tiles are the window's 8 frames in time order (numbered), cropped around
both players with some room, Player 1 outlined blue and Player 2 orange.

usage: python make_grids.py <name>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
TILE_H = 360
COLOUR = [(201, 99, 37), (31, 119, 217)]  # BGR: Player 1 blue, Player 2 orange


def main() -> int:
    name = sys.argv[1]
    data = json.loads((HERE / f"windows_{name}.json").read_text(encoding="utf-8"))
    out_dir = HERE / "clips" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(data["video"])
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pos, cache = 0, {}
    for wi, w in enumerate(data["windows"]):
        for f in w["frames"]:
            if f in cache:
                continue
            while pos < f:
                cap.grab(); pos += 1
            ok, img = cap.read(); pos += 1
            cache[f] = img if ok else np.zeros((H, W, 3), np.uint8)
        bx = np.array([b for pl in w["boxes"] for b in pl if b], dtype=float)
        if len(bx):
            x0, y0, x1, y1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
            h = y1 - y0
            X0, X1 = int(max(0, x0 - 0.35 * h)), int(min(W, x1 + 0.35 * h))
            Y0, Y1 = int(max(0, y0 - 0.12 * h)), int(min(H, y1 + 0.08 * h))
        else:
            X0, Y0, X1, Y1 = 0, 0, W, H
        tiles = []
        for t, f in enumerate(w["frames"]):
            img = cache[f].copy()
            for k in range(2):
                b = w["boxes"][k][t]
                if b:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLOUR[k], 4)
            crop = img[Y0:Y1, X0:X1]
            tw = max(1, round(crop.shape[1] * TILE_H / max(1, crop.shape[0])))
            tile = cv2.resize(crop, (tw, TILE_H))
            cv2.rectangle(tile, (0, 0), (34, 30), (0, 0, 0), -1)
            cv2.putText(tile, str(t + 1), (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            tiles.append(tile)
        tw = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, tw - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
        scale = min(1.0, 1100 / max(grid.shape[:2]))
        grid = cv2.resize(grid, (round(grid.shape[1] * scale), round(grid.shape[0] * scale)))
        fn = f"{wi:04d}_{w['start_s']:.1f}s.jpg"
        cv2.imwrite(str(out_dir / fn), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        w["image"] = f"clips/{name}/{fn}"
        for f in [f for f in cache if f < w["frames"][0]]:
            del cache[f]
    (HERE / f"windows_{name}.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{name}: {len(data['windows'])} grids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
