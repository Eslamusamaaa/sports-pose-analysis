# Setup — from `git clone` to a running system

Tested environment: Windows 11, NVIDIA RTX 3050 Laptop (4 GB), no CUDA toolkit, no C++ compiler.
All commands run from the **repository root** unless stated.

## 0. Layout the scripts expect

```
sports-pose-analysis/
├── app/                 web app (this repo)
├── swimming/  karate_vlm/  docs/  tools/
├── mmaction2-main/      cloned by step 2   (gitignored, read-only dependency)
├── openpose-master/     cloned by step 2   (gitignored, only the prototxt is used)
├── external/            weights + pose libs (gitignored)
├── data/                datasets you download (gitignored)
└── work_dirs/           training output (gitignored)
```

## 1. Prerequisites
Python 3.12 (backend) **and** Python 3.11 (MMAction2 venv), Node 20+, `git`, `ffmpeg`/`ffprobe` on PATH, an NVIDIA driver supporting CUDA 11.8+.

## 2. Web app

```powershell
git clone https://github.com/Eslamusamaaa/sports-pose-analysis.git
cd sports-pose-analysis

# external sources + RTMPose/YOLO11 weights + pose libs (mmaction2-main, openpose-master, external/)
powershell -ExecutionPolicy Bypass -File app/scripts/download_external.ps1

cd app
Copy-Item .env.example .env
python -m pip install -r backend/requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_mmaction_env.ps1     # Python 3.11 venv: torch 2.1 + mmcv 2.1 + mmaction2
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1        # BODY_25 caffemodel + NTU60 / HMDB / UCF / K400 checkpoints
powershell -ExecutionPolicy Bypass -File scripts/setup_openpose_gpu.ps1     # GPU OpenPose 1.7.0 build (needs OPENPOSE_GPU_DIR in .env)
```

Run:

```powershell
cd backend ; python -m uvicorn app.main:app --port 8000       # terminal 1
cd ../frontend ; npm install ; npm run dev                    # terminal 2  → http://localhost:5173
```

The trained swimming checkpoint ships in the repo (`app/models/swim/`), so the Swimming page works without training.
`GET /api/health` reports which parts are ready (OpenPose build, MMAction2 venv, checkpoints, pose engines).

Optional: CUDA venv for MMAction2 (`scripts/setup_mmaction_cuda_D.ps1`, installs on another drive and then set `MMACTION_PYTHON` in `.env`);
OSNet re-ID weights `osnet_ain_x1_0_msmt17.pth` → `app/backend/reid/` (see `download_external.ps1` header).

## 3. Reproduce the swimming model

```powershell
# 3.1 data: SwimXYZ (CC-BY 4.0, https://zenodo.org/records/8399376; paper DOI 10.1145/3623264.3624440), ~6.6 GB, md5-checked
mkdir data/swimxyz/raw ; copy swimming/download.sh and swimming/manifest.txt into it ; run `bash download.sh` there (Git Bash)

# 3.2 SwimXYZ → MMAction2 PoseDataset (COCO-17, 30 fps, 60-frame windows), one pickle per stroke
python swimming/scripts/convert_swimxyz.py --strokes freestyle   --out data/swimxyz/part_freestyle.pkl
python swimming/scripts/convert_swimxyz.py --strokes backstroke  --out data/swimxyz/part_backstroke.pkl
python swimming/scripts/convert_swimxyz.py --strokes breaststroke --out data/swimxyz/part_breaststroke.pkl
python swimming/scripts/convert_swimxyz.py --strokes butterfly   --out data/swimxyz/part_butterfly.pkl

# 3.3 leakage-aware split (by underlying motion)
python swimming/scripts/make_split.py --src data/swimxyz/part_freestyle.pkl data/swimxyz/part_backstroke.pkl data/swimxyz/part_breaststroke.pkl data/swimxyz/part_butterfly.pkl --dst data/swimxyz/swimxyz_coco17_split.pkl

# 3.4 train (use the MMAction2 venv; config paths are relative to the repo root)
$env:PYTHONPATH = "swimming/scripts"
app/third_party/mmaction-venv/Scripts/python.exe mmaction2-main/tools/train.py swimming/scripts/stgcn_swim4.py

# 3.5 evaluate (per view, per stroke, clean vs corrupted) → swimming/reports/eval_report.json
app/third_party/mmaction-venv/Scripts/python.exe swimming/scripts/evaluate.py --checkpoint work_dirs/swim/best_acc_top1_epoch_9.pth
```

Copy the best checkpoint to `app/models/swim/stgcn_swim4.pth` to use it in the web app.
`swimming/logs/train.log` is the original run (10 epochs, best 99.93 % at epoch 9 on synthetic held-out motions).

## 4. Karate VLM experiment

```bash
bash karate_vlm/download_models.sh        # llama.cpp (CUDA) + 3 GGUF VLMs into external/vlm  (~10 GB)
```

1. Analyse a match in the web app (gives a job under `app/data/jobs/<job_id>/`).
2. `python karate_vlm/scripts/windows.py <job_id> <name>` — windows + player boxes (`windows_<name>.json`).
3. `make_grids.py <name>` (in the MMAction2 venv, needs OpenCV) — 2×4 frame grids.
4. `python karate_vlm/scripts/run_vlm.py qwen3vl` (or `qwen35`, `gemma3`) — labels every window.
5. `report.py <model> <name>` / `compare.py <name>` — HTML reports.

The scripts read/write next to themselves, so copy them to a working folder (or run from `karate_vlm/scripts/`) and keep `windows_*.json`, `clips/` and `results_*.json` together.
Set `KARATE_VLM_DIR` to move the VLM binaries/models, and `KARATE_EXTRA_JOB_DIRS` (`;`-separated) for jobs stored on another drive.
Reference outputs from the original run are in `karate_vlm/results/`.

## 5. Known gaps
- No input videos are included (rights unclear) — bring your own.
- OSNet re-ID weights are a manual download (the app runs without them).
- The OpenPose GPU build is fetched by `setup_openpose_gpu.ps1` from the official GitHub release (418 MB, size is asserted).
- Karate has no trained classifier yet; see `docs/STATUS.md`.
