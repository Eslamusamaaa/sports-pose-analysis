"""Players-only overlay video: "Player 1" / "Player 2" on the people that matter.

The players are the tracks the converter already chose for the action model:
ranked by confidence x presence x MOTION, which is what keeps referees and
spectators out (see skeleton_service.track_people). Labelling exactly those
tracks means the names on screen and the skeletons the model classified are
the same people. Swimming uses one slot (the stitched swimmer).

The engine's own overlay (every detected person) is replaced in the UI by this
render; if rendering fails the original overlay stays, so the job never loses
its skeleton video.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import BACKEND_DIR, get_settings
from ..utils.logging_utils import JobLogger
from . import video_service

SCRIPT = BACKEND_DIR / "render_players.py"


def labels_for(n: int) -> list[str]:
    return [f"Player {i + 1}" for i in range(n)]


def render(parsed: dict, player_ids: list[int], video: Path, players_json: Path,
           out_avi: Path, out_mp4: Path, log: JobLogger) -> dict:
    s = get_settings()
    slot_of = {tid: i for i, tid in enumerate(player_ids)}
    frames = []
    for f in parsed["frames"]:
        row = []
        for p in f["people"]:
            if p.get("interpolated"):
                continue  # filled-in skeleton for the action model only; not drawn
            # Everyone is drawn (detection of all persons); non-players get
            # slot -1 = thin grey box and skeleton, no label.
            slot = slot_of.get(p.get("track_id"), -1)
            row.append({
                "slot": slot,
                "kp": [v for joint in p["keypoints"] for v in joint],
                "bbox": p.get("bbox"),
            })
        frames.append(row)
    labels = labels_for(len(player_ids))
    players_json.parent.mkdir(parents=True, exist_ok=True)
    players_json.write_text(json.dumps({"labels": labels, "frames": frames}), encoding="utf-8")

    proc = subprocess.run(
        [str(s.active_mmaction_python), str(SCRIPT), str(video), str(players_json), str(out_avi)],
        cwd=str(BACKEND_DIR), capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or not out_avi.exists():
        raise RuntimeError(f"player overlay render failed: {(proc.stderr or '')[-800:]}")
    if not video_service.to_browser_mp4(out_avi, out_mp4, log):
        raise RuntimeError("player overlay mp4 transcode failed")
    present = {lbl: sum(1 for row in frames if any(p["slot"] == i for p in row))
               for i, lbl in enumerate(labels)}
    log.info(
        "players overlay: " + ", ".join(
            f"{lbl} = track {tid} ({present[lbl]} frames)" for lbl, tid in zip(labels, player_ids)
        ) + "; everyone else is drawn in grey without a label",
        stage="skeleton",
    )
    return {"labels": labels, "track_ids": player_ids, "frames_present": present}
