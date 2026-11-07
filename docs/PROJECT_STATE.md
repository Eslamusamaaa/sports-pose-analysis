# Karate Action Recognition — Project Handoff Prompt

> Copy everything below this line into a new session / hand to another engineer.

---

## CONTEXT

I have a working end-to-end computer-vision pipeline for analysing karate video.
It runs locally on Windows 11 with an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
**The infrastructure is complete and verified. What is missing is a karate-trained
classifier — and the labelled data required to train one.**

Do not rebuild the pipeline. Help me with the remaining gap.

---

## 1. WHAT IS ALREADY BUILT AND WORKING

### Architecture

```
Browser (React/Vite/TS)
    │  HTTP + WebSocket  (Vite proxies /api, ws:true)
    ▼
FastAPI backend  (system Python 3.12)
    │  subprocess
    ├──► OpenPose 1.7.0 GPU binary ──► BODY_25 keypoints (JSON per frame)
    │
    ├──► centroid tracker ──► temporally consistent person tracks
    │
    ├──► BODY_25 (25 joints) ──► COCO-17 (17 joints) conversion
    │
    └──► MMAction2 workers  (separate Python 3.11 venv, torch 2.1.0+cu118)
             └──► ST-GCN / PoseC3D ──► action scores
```

### Repo layout

```
karate-ai-web/
├── backend/
│   ├── app/
│   │   ├── api/routes.py              REST + WebSocket endpoints
│   │   ├── config.py                  .env-driven settings, device resolution
│   │   ├── jobs.py                    job registry + on-disk artefact layout
│   │   ├── main.py                    FastAPI entrypoint
│   │   ├── pipelines/video_analysis.py   stage-isolated orchestrator
│   │   ├── services/
│   │   │   ├── openpose_service.py    subprocess wrapper for OpenPoseDemo.exe
│   │   │   ├── skeleton_service.py    BODY_25 JSON parsing + motion-weighted tracking
│   │   │   ├── skeleton_converter.py  BODY_25 → COCO-17 (documented, lossless subset)
│   │   │   ├── keypoints.py           both joint layouts + mapping table
│   │   │   ├── timeline_service.py    sliding-window classification over whole clip
│   │   │   ├── ensemble_service.py    multi-model evidence pooling
│   │   │   ├── model_zoo.py           model registry + concept mapping
│   │   │   ├── mmaction_service.py    one-shot inference
│   │   │   ├── realtime_service.py    live webcam session manager
│   │   │   └── video_service.py       ffprobe/ffmpeg validate, trim, transcode
│   │   └── utils/logging_utils.py     per-job logging (file + memory + stdout)
│   ├── mmaction_worker.py             PERSISTENT worker (model loaded once)
│   └── mmaction_infer.py              one-shot worker
├── frontend/src/components/
│   ├── UploadPanel.tsx  PipelineStatus.tsx  OpenPoseResult.tsx
│   ├── MMActionResult.tsx  TimelinePanel.tsx  LogPanel.tsx
│   ├── LiveView.tsx  SkeletonCanvas.tsx  JobPicker.tsx
├── scripts/
│   ├── setup_mmaction_env.ps1         CPU venv (Python 3.11)
│   ├── setup_mmaction_cuda_D.ps1      CUDA venv on D: drive
│   ├── setup_openpose_gpu.ps1         GPU OpenPose install + verification
│   └── download_models.ps1            model weights
└── data/jobs/<job_id>/                per-job artefacts
```

### Verified working features

1. **Batch video analysis** — upload video → full pipeline → results in browser.
2. **Live webcam detection** — OpenPose owns the camera, streams over WebSocket,
   side-by-side view: clean camera frame (left) + BODY_25 skeleton overlaid on the
   same frame (right), with rolling classification.
3. **Sliding-window timeline** — classifies overlapping 1-second windows so
   **every frame is covered**, producing "from 21.0s to 22.0s: X" output.
4. **Multi-model ensemble** — several checkpoints classify the same window;
   their martial-arts-relevant classes are mapped to shared concepts
   (punch / kick / grapple / martial / move) and pooled by "lift over uniform
   baseline", with a vote count per concept.
5. **Job browser** — reopen any previous analysis without re-uploading.
6. **Stage isolation** — MMAction2 failure yields status `partial`, never
   invalidating valid OpenPose output. Real subprocess stderr surfaces in the UI.

### Measured performance (real numbers from this machine)

| Stage | CPU | GPU |
|---|---|---|
| OpenPose BODY_25 | 2.19 s/frame (0.46 fps) | **0.071–0.096 s/frame (~11–14 fps)** |
| ST-GCN, one-shot subprocess | 4.87 s | 1.18 s |
| ST-GCN, persistent worker | 0.09–0.17 s | **0.012–0.014 s** |
| Full 4865-frame 1080p video | ~3 hours | **~18.5 min** |
| Timeline, 324 windows × 1 model | 4.4 s | — |
| Ensemble, 29 windows × 2 models | — | 36 s |

### Key technical decisions (already solved — do not redo)

- **BODY_25 → COCO-17 is mandatory.** The ST-GCN config declares
  `graph_cfg=dict(layout='coco')` which resolves to `num_node=17`, while OpenPose
  emits 25 joints. Every COCO joint has an exact BODY_25 counterpart, so the
  conversion is a pure gather (subset + reorder), no interpolation. 8 joints are
  dropped: Neck, MidHip, and 6 foot joints (toes/heels).
- **Coordinates stay in raw pixels.** `PreNormalize2D` inside MMAction2 applies
  `(x - w/2)/(w/2)`. Normalising beforehand would double-normalise.
- **OpenPose does NO tracking** (`person_id` is always `[-1]`). A greedy
  nearest-centroid tracker assigns stable ids. Ranking is
  `confidence × frames_present × motion` — motion is essential: with
  confidence×presence alone the system selected stationary **referees** (5.6 px/frame)
  over the actual athletes (16–21 px/frame).
- **Separate Python venv for MMAction2** — `mmcv` has no wheel for Python 3.12 /
  torch 2.13, and this machine has no C++ compiler. Runs as a subprocess, which
  also isolates crashes from the API.
- **`DecompressPose` must be stripped** from PoseC3D test pipelines; it expects
  the packed annotation format, not live `inference_skeleton` input.
- **Window length matters enormously** (measured on real karate footage):

  | window | seconds | strikes detected |
  |---|---|---|
  | 100 frames | 3.34 s | 1 |
  | **30 frames** | **1.00 s** | **6** ⭐ |
  | 12 frames | 0.40 s | 0 |

  There is an optimum near 1 second — shorter is NOT better, because the model
  needs enough temporal context.

---

## 2. CURRENT CLASSIFICATION RESULTS (the honest part)

Models currently in the ensemble: **ST-GCN (NTU RGB+D 60)** and
**PoseC3D (NTU RGB+D 60)**.

On a real karate kumite match (4865 frames, 1080p, 162 s):

- Both models **independently agreed** on `punching/slapping other person`
  at **t = 22 s** (0.569 and 0.645) — different architectures (GCN vs 3D-CNN)
  reaching the same label is the strongest signal obtained so far.
- Across the whole clip, ~88% of windows classify as `walking towards each other`.
  This is arguably *correct*: most of a match is circling and footwork.
- Only ~6 windows out of 324 surfaced a strike class.

**These models were NOT trained on karate.** NTU RGB+D 60 contains 60 everyday
indoor actions; `punching/slapping other person` is a casual scuffle, not a
*tsuki*. The system therefore cannot name a technique and never claims to.

### Models tested and REJECTED (with evidence)

HMDB51, UCF101 and Kinetics-400 PoseC3D checkpoints were downloaded and tested.
On this footage they returned **near-random labels**: `HulaHoop`, `MoppingFloor`,
`riding unicycle`, `catching or throwing frisbee`, `robot dancing`. Pooling them
pushed a meaningless `martial-arts-like` concept (fed by `TaiChi`/`capoeira`)
above `punch` in the rankings. **Diagnosis:** they are trained on single-subject
close-framed clips, and PoseC3D rasterises the skeleton into a 64×64 heatmap —
a 1080p wide shot with two distant athletes plus officials is far outside their
distribution. They remain in the registry but are disabled by default.

---

## 3. WHAT IS MISSING — THE ACTUAL GAP

### 3.1 Labelled karate data (the blocker)

There is **no public, downloadable, labelled karate VIDEO dataset** suitable for
this 2D pipeline. Verified:

| Dataset | Downloadable | Action labels | 2D video |
|---|:--:|:--:|:--:|
| Karate Kumite 35K+ (Bern thesis) | ❌ paper only | ✅ | ✅ |
| Karate Technical Action (1,847 clips) | ❌ paper only | ✅ | ✅ |
| MS-KARD (1,518 clips, 23 classes) | ❌ "intended public" | ✅ | ✅ |
| MADS (martial arts/dance/sports) | ⚠️ site down | ❌ *3D joint coords only, 6 karate sequences* | ⚠️ |
| Kyokushin MoCap (3,229 strikes) | ✅ Figshare | ✅ | ❌ 3D MoCap |
| Karate Stances 2026 | ✅ CC-BY 4.0 | ⚠️ stances only | ❌ |
| Punch_DL | ✅ GitHub | ✅ | ⚠️ only ~240 punches |

**Note on MADS specifically:** it is a *3D pose-estimation benchmark*, not an
action-recognition dataset. Its ground truth is per-frame 3D joint coordinates
from optical MoCap; it contains **no action class labels at all**, and only
6 karate sequences. It does not fit this use case.

### 3.2 What still needs building

1. **Annotation tool** — UI to scrub a video, mark `[start_s, end_s] → label`,
   and persist. This is the only path to ground truth and nothing can substitute
   for it.
2. **Dataset builder** — convert annotated skeleton segments into MMAction2
   `PoseDataset` pickle format:
   ```python
   {'frame_dir': str, 'label': int,
    'keypoint': ndarray (M, T, 17, 2),      # raw pixels
    'keypoint_score': ndarray (M, T, 17),
    'img_shape': (h, w), 'total_frames': T}
   ```
3. **Training config + script** — copy the NTU60 ST-GCN config, set
   `num_classes`, point `ann_file` at the new pickle, and set
   `load_from` to the NTU60 checkpoint so pretrained motion features are kept.
4. **Evaluation** — **split by performer, never randomly.** A random split puts
   clips of the same athlete in both train and test, so the model memorises bodies
   and reports inflated accuracy that collapses on a new athlete. Report per-class
   accuracy and a confusion matrix on held-out performers.
5. **Stronger tracker for kumite** — the current nearest-centroid tracker has no
   re-identification and no occlusion handling. When two fighters clinch, identities
   swap and the skeleton sequence becomes garbage. ByteTrack or OC-SORT is needed.
   (Kata is single-person and does not have this problem.)

### 3.3 Known limitations to preserve

- Dropping the 6 BODY_25 foot joints loses information relevant to distinguishing
  kick types (*mawashi-geri* vs *mae-geri*). A custom 25-node `graph_cfg` trained
  directly on BODY_25 would keep them, but forfeits NTU60 pretrained weights.
- 4 GB VRAM: running live detection and a batch job simultaneously causes GPU
  contention and slows OpenPose by ~10×. They must not run at once.
- Timings are exact and person selection is correct; **the labels are the weak part.**

---

## 4. WHAT I WANT HELP WITH

Pick up from here. Priorities:

1. Build the annotation tool so I can create ground truth from my own match footage.
2. Build the dataset builder + training config + honest evaluation.
3. Advise on obtaining one of the gated karate datasets (who to contact, what to ask).
4. Upgrade the tracker for kumite (two fighters, frequent occlusion).

**Constraints:** Windows 11, RTX 3050 4 GB, C: nearly full (use D:), no C++
compiler available, OpenPose and MMAction2 are read-only external dependencies.

**Rule:** never describe the system as recognising karate until it has been
trained AND evaluated on held-out karate performers.
