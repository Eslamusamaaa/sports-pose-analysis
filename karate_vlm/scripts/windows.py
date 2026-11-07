"""Step 1 (backend python): split the whole video into overlapping windows.

Windows of WIN_S seconds every STEP_S seconds, from the first frame to the
last. Players come from the same identification as the app (Player 1 /
Player 2). For each window: the sampled frames and each player's box in
them (None where that player was not detected). Writes windows_<name>.json.

usage: python windows.py <job_id> <name>
"""
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "karate-ai-web" / "backend"
sys.path.insert(0, str(BACKEND))

from app.services import player_follow, skeleton_service  # noqa: E402
from app.utils.logging_utils import JobLogger  # noqa: E402

import os
JOB_DIRS = [BACKEND.parent / "data" / "jobs"] + [Path(d) for d in os.environ.get("KARATE_EXTRA_JOB_DIRS", "").split(os.pathsep) if d]
WIN_S = 1.2     # window length
STEP_S = 0.8    # a new window every 0.8 s (0.4 s overlap, so no technique is cut in two)
N_FRAMES = 8    # frames sent to the model per window


def main() -> int:
    job_id, name = sys.argv[1], sys.argv[2]
    job_dir = next(d / job_id for d in JOB_DIRS if (d / job_id).exists())
    meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    fps = (meta.get("video_meta") or {}).get("fps") or 30.0
    log = JobLogger(job_id + "_vlm", Path(__file__).parent / "logs")
    parsed = skeleton_service.parse_openpose_dir(job_dir / "openpose" / "json", log)
    parsed = skeleton_service.track_people(parsed, log, rank_by="limbs")
    player_follow.follow_players(parsed, 2, log, fps=fps)
    frames = parsed["frames"]
    n = len(frames)

    def box(i, k):
        p = next((p for p in frames[i]["people"]
                  if p.get("player_slot") == k and not p.get("interpolated")), None)
        return player_follow._box(p) if p else None

    win, step = round(WIN_S * fps), round(STEP_S * fps)
    windows = []
    for s in range(0, max(1, n - win + 1), step):
        idx = [round(s + t * (win - 1) / (N_FRAMES - 1)) for t in range(N_FRAMES)]
        windows.append({"start_s": round(s / fps, 2), "end_s": round((s + win) / fps, 2), "frames": idx,
                        "boxes": [[box(i, 0) for i in idx], [box(i, 1) for i in idx]]})
    out = {"job": job_id, "name": name, "fps": fps, "frames_total": n,
           "video": str(job_dir / "input.mp4"), "windows": windows}
    Path(__file__).with_name(f"windows_{name}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{name}: {len(windows)} windows over {n / fps:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
