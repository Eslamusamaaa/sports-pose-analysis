# karate-ai-web

Proof of concept proving one thing end to end:

```
Browser → FastAPI backend → OpenPose (BODY_25) → skeleton JSON
        → BODY_25→COCO-17 conversion → MMAction2 ST-GCN → prediction → Browser
```

Both source repositories (`openpose-master/`, `mmaction2-main/`) are treated as
**external, read-only dependencies**. Nothing in this project writes to either one.

> **This system does not recognise karate.** The bundled checkpoint is ST-GCN
> pretrained on **NTU RGB+D 60**, a dataset of 60 everyday indoor actions with no
> karate classes. Any prediction on karate footage is **out-of-domain** and is
> labelled as such throughout the UI and API. See
> [Next step: making it recognise karate](#17-next-step-making-it-recognise-karate).

---

## 1. Project structure

```
karate-ai-web/
├── backend/
│   ├── app/
│   │   ├── api/routes.py             REST + WebSocket endpoints
│   │   ├── pipelines/video_analysis.py   batch orchestrator (stage-isolated)
│   │   ├── services/
│   │   │   ├── openpose_service.py       subprocess wrapper for OpenPoseDemo.exe
│   │   │   ├── skeleton_service.py       BODY_25 JSON parsing + centroid tracking
│   │   │   ├── skeleton_converter.py     BODY_25 → COCO-17 (documented)
│   │   │   ├── keypoints.py              both joint layouts + the mapping table
│   │   │   ├── mmaction_service.py       one-shot MMAction2 invocation
│   │   │   ├── realtime_service.py       live webcam session manager
│   │   │   └── video_service.py          ffprobe/ffmpeg validate, trim, transcode
│   │   ├── utils/logging_utils.py    per-job logging (file + memory + stdout)
│   │   ├── config.py                 .env-driven configuration
│   │   ├── jobs.py                   job registry + on-disk layout
│   │   └── main.py                   FastAPI entrypoint
│   ├── mmaction_infer.py             batch worker, runs INSIDE the mmaction venv
│   ├── mmaction_worker.py            persistent worker for real-time
│   └── requirements.txt
├── frontend/                         React + Vite + TypeScript
│   └── src/components/               LiveView, UploadPanel, PipelineStatus, …
├── data/
│   ├── jobs/<job_id>/                per-job artefacts (see §13)
│   └── live/                         live-session scratch (auto-pruned)
├── models/
│   ├── openpose/pose/body_25/        prototxt + caffemodel
│   └── mmaction/                     checkpoint + label map
├── scripts/
│   ├── setup_mmaction_env.ps1        builds the isolated Python 3.11 venv
│   ├── download_models.ps1           fetches all weights
│   └── setup_openpose_gpu.ps1        installs the GPU build + CUDA DLLs
├── third_party/                      OpenPose builds + mmaction venv (gitignored)
├── .env.example
└── README.md
```

---

## 2. Environment this was built and verified on

| Item | Value |
|---|---|
| OS | Windows 11 Pro, build 22621 |
| CPU Python | 3.12.10 (backend) |
| MMAction2 Python | 3.11.9 (isolated venv) |
| GPU | NVIDIA RTX 3050 Laptop, 4 GB VRAM |
| Driver | 572.47 (supports CUDA 12.8) |
| CUDA toolkit | **not installed** — no `nvcc` |
| C++ compiler | **none** — no MSVC, no CMake |
| ffmpeg / ffprobe | present |
| Node / npm | 24.19.0 / 11.17.0 |

Two consequences drove the whole design:

1. **Nothing can be compiled.** OpenPose could not be built from source, and
   `mmcv` had to come from a prebuilt wheel.
2. **`mmcv` has no wheel for Python 3.12 / torch 2.13.** The newest combination
   with a prebuilt Windows wheel is **Python 3.11 + torch 2.1.0**, so MMAction2
   lives in its own venv and is driven over a subprocess. This also isolates
   MMAction2 crashes from the API.

---

## 3. Setup

```powershell
cd karate-ai-web
Copy-Item .env.example .env
```

### 3.1 Backend dependencies (system Python 3.12)

```bash
python -m pip install -r backend/requirements.txt
```

### 3.2 MMAction2 environment (isolated Python 3.11 venv)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_mmaction_env.ps1
```

Installs torch 2.1.0+cpu, mmengine, mmcv 2.1.0 (prebuilt wheel), then links the
**local** `mmaction2-main` checkout via a `.pth` file.

Two real packaging bugs this script works around:

* The **PyPI `mmaction2` 1.2.0 wheel is broken** — it omits
  `mmaction/models/localizers/drn/`, so `import mmaction.models` raises
  `ModuleNotFoundError: No module named 'mmaction.models.localizers.drn'`. The
  local checkout has it, so the venv points at the repo through a `.pth` file.
  Unlike `pip install -e .`, this writes **nothing** into the repo.
* **numpy must stay below 2.** `opencv-python` 5.x and `matplotlib` pull numpy
  2.x, which breaks torch 2.1.0 (`Failed to initialize NumPy: _ARRAY_API not
  found`) and mmcv 2.1.0, both compiled against the numpy 1.x C ABI. numpy is
  therefore pinned and installed **last**, and opencv is pinned to 4.10.0.84.

### 3.2b Optional: CUDA MMAction2 venv (installed on D:)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_mmaction_cuda_D.ps1
```

Installs torch 2.1.0**+cu118** and the prebuilt **cu118** mmcv wheel into
`D:\Users\karate-ai-cuda\mmaction-venv`, then point `.env` at it:

```
MMACTION_PYTHON=D:/Users/karate-ai-cuda/mmaction-venv/Scripts/python.exe
```

Why D: — the cu118 torch wheel is **2.7 GB** and expands to ~4 GB; peak usage
during install is ~6.7 GB, which does not fit on this C: drive. The script also
redirects `PIP_CACHE_DIR`, `TMP` and `TEMP` onto D:, because otherwise pip still
stages the 2.7 GB download on C: and fills it regardless. `D:\` root rejected
writes without elevation, so the script probes for the first writable location
(`D:\Users`, then `D:\Downloads`) rather than assuming the drive root.

**Honest assessment: this buys very little.** ST-GCN inference is already
0.09–0.17 s per window via the persistent worker, and CUDA takes it to roughly
0.02 s. OpenPose dominates total runtime by two orders of magnitude, so the GPU
OpenPose build (§3.4) is what actually delivers real-time.

**Venv readiness marker.** Both setup scripts write `.karate-venv-ready` into the
venv as their *final* step, and the backend refuses to select a venv without it.
This matters because torch installs first and `numpy<2` is re-pinned last; in
between, the venv imports torch but fails with `Failed to initialize NumPy:
_ARRAY_API not found`. Without the marker the backend would switch to a
half-built environment mid-install. If the configured venv is not ready, the
backend automatically falls back to the CPU venv and reports
`using_cpu_fallback_venv: true` from `/api/health`.

### 3.3 Model weights

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
```

> OpenPose's bundled `models/getBaseModels.bat` points at
> `posefs1.perception.cs.cmu.edu`, which is **offline**. The script uses a
> public mirror and asserts the exact byte count instead.

### 3.4 OpenPose

The portable **CPU** build is the fallback. For real-time you need the GPU build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_openpose_gpu.ps1
```

The script reads `OPENPOSE_GPU_DIR` from `.env` and **extracts the zip directly
to that location**. The backend then auto-prefers the GPU build
(`OPENPOSE_PREFER_GPU=true`) and silently falls back to CPU if it is missing.

Two hard-won details encoded in the script:

* **Never relocate the extracted build across volumes.** A `Move-Item` from C:
  to D: corrupted a working build: every filename and every file *size* still
  matched, but `OpenPoseDemo.exe` died at DLL-load with `0xC000012F`
  (`STATUS_INVALID_IMAGE_NOT_MZ`) printing **nothing at all**. Copying it back to
  C: failed identically, proving the bytes — not the drive — were the problem.
* **The zip size is asserted exactly** (437,912,637 bytes) and downloads resume
  with `-C -`. This 418 MB GitHub asset truncates fairly often (curl exit 18),
  and a truncated zip yields DLLs that look correct and fail to load.

The script finishes by *running* the binary and checking it prints its flags,
because file inspection alone cannot distinguish a sound build from a corrupt one.

No CUDA toolkit and no cuDNN account are required: the 1.7.0 GPU release already
bundles `cublas64_11`, `cublasLt64_11`, `cudart64_110` and `curand64_10`, and it
runs without the cuDNN sub-libraries.

---

## 4. Running it

**Backend** (single worker — the job registry is in-process):

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to `127.0.0.1:8000`, so the
browser only ever talks to one origin.

---

## 5. Performance: CPU vs GPU

The prebuilt **CPU** OpenPose runs BODY_25 at a measured **2.19 s/frame
(~0.46 fps)** at `-1x256` on this machine. That is the single dominant cost in
the pipeline, and it is why real-time needs the GPU build.

| Stage | CPU build | GPU build |
|---|---|---|
| OpenPose BODY_25 | **2.19 s/frame** | see §14 |
| Skeleton parse + tracking | < 0.1 s total | same |
| MMAction2 ST-GCN (one-shot subprocess, `num_clips=10`) | 4.87 s | — |
| MMAction2 ST-GCN (**persistent worker**, `num_clips=1`) | **0.09–0.17 s** | — |

The persistent worker (`backend/mmaction_worker.py`) loads the recognizer once
and answers over stdin/stdout, which is a **~50× improvement** and takes
MMAction2 off the critical path entirely. Putting ST-GCN on CUDA would save
roughly a further 0.1 s — negligible next to OpenPose — so GPU effort goes to
OpenPose first.

`num_clips=1` vs the config's `10` trades temporal averaging for ~10× lower
latency. Every response reports the value used, so the UI never implies
full-accuracy inference.

---

## 6. Environment variables

See `.env.example` for the annotated list. The ones that matter most:

| Variable | Default | Meaning |
|---|---|---|
| `OPENPOSE_DIR` / `OPENPOSE_BIN` | `third_party/openpose` | CPU build |
| `OPENPOSE_GPU_DIR` | `third_party/openpose-gpu` | GPU build; **auto-preferred when present** |
| `OPENPOSE_PREFER_GPU` | `true` | set `false` to force the CPU build |
| `OPENPOSE_MODEL_DIR` | `models/openpose` | must contain `pose/body_25/…` |
| `OPENPOSE_NET_RESOLUTION` | `-1x256` | lower = faster, less accurate |
| `OPENPOSE_MAX_FRAMES` | `600` | frame cap, **CPU build** (0 = unlimited) |
| `OPENPOSE_MAX_FRAMES_GPU` | `1800` | frame cap, **GPU build** |
| `MMACTION_PYTHON` | `third_party/mmaction-venv/Scripts/python.exe` | venv interpreter |
| `MMACTION_CONFIG` | ST-GCN NTU60 xsub 2D | declares `layout='coco'` |
| `CHECKPOINT_PATH` / `LABEL_MAP_PATH` | `models/mmaction/…` | weights + 60 labels |
| `DEVICE` | `cuda:0` | MMAction2 device; falls back to CPU and **reports which was used** |
| `REALTIME_WINDOW` | `100` | sliding-window length for live ST-GCN |
| `REALTIME_INFER_EVERY` | `10` | run inference every N frames |
| `REALTIME_NUM_CLIPS` | `1` | latency/averaging tradeoff |
| `REALTIME_FRAME_HW` | `480x640` | camera frame size for `PreNormalize2D` |

The frame cap is **device-aware**: `effective_max_frames` returns the GPU cap
when the GPU build is installed, so a cap that is comfortable on GPU can never
silently trigger a multi-hour CPU run.

---

## 7. The OpenPose command

Exactly what the backend executes (`services/openpose_service.py`):

```
<OPENPOSE_BIN>
  --video          <job>/input.mp4
  --model_folder   <OPENPOSE_MODEL_DIR>/       # trailing separator required
  --model_pose     BODY_25
  --net_resolution -1x256
  --write_json     <job>/openpose/json
  --write_video    <job>/openpose/output_video.avi
  --display        0
  --render_pose    1
  --frame_last     <cap-1>                     # inclusive, hence -1
```

Three things learned from the real binary, all encoded in the service:

* It **must** run with `cwd` = the OpenPose root. From `bin/` it cannot find its
  resources.
* `--model_folder` **must end with a separator**, and forward slashes are used
  to avoid the Windows backslash-before-quote escaping trap. (A trailing `\"`
  swallowed the rest of the argument list during development.)
* `--frame_last N` is **inclusive** → yields N+1 frames.

Output is **MJPG-in-AVI**, which no browser `<video>` can play, so
`video_service.to_browser_mp4` transcodes to H.264 for the UI.

---

## 8. OpenPose JSON format (verified, not assumed)

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [-1],
      "pose_keypoints_2d": [x0, y0, c0, x1, y1, c1, ...],
      "face_keypoints_2d": [], "hand_left_keypoints_2d": [], ...
    }
  ]
}
```

* `pose_keypoints_2d` is a **flat list of 75 floats** = 25 joints × (x, y, confidence),
  in **pixel** coordinates. Undetected joints are exactly `(0, 0, 0)`.
* `person_id` is **always `[-1]`** — OpenPose performs **no cross-frame identity
  association**, and people arrive in an arbitrary, frame-varying order.

That second point matters: feeding raw per-frame ordering to a GCN scrambles
each person slot over time. `skeleton_service.track_people` therefore runs a
greedy nearest-centroid tracker to build temporally consistent tracks, and the
top-scoring tracks fill the person slots. It is deliberately simple — no
re-identification, no occlusion handling — and long gaps start a new track.

The raw OpenPose JSON is **never modified**; parsed output is written alongside it.

---

## 9. Keypoint formats and the conversion

**BODY_25 is not compatible with this checkpoint.** The config

```
configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
  backbone = dict(type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial'))
  cls_head = dict(type='GCNHead', num_classes=60, in_channels=256)
```

resolves `layout='coco'` in `mmaction/models/utils/graph.py` to **`num_node = 17`**.
OpenPose emits **25** joints in a different order — a hard shape mismatch.

`services/skeleton_converter.py` implements the mapping. Every COCO-17 joint has
an **exact** BODY_25 counterpart, so this is a pure **gather (subset + reorder)**.
**No joint is interpolated, averaged or invented.**

| COCO | name | ← BODY_25 | name |
|---|---|---|---|
| 0 | nose | 0 | Nose |
| 1 | left_eye | 16 | LEye |
| 2 | right_eye | 15 | REye |
| 3 | left_ear | 18 | LEar |
| 4 | right_ear | 17 | REar |
| 5 | left_shoulder | 5 | LShoulder |
| 6 | right_shoulder | 2 | RShoulder |
| 7 | left_elbow | 6 | LElbow |
| 8 | right_elbow | 3 | RElbow |
| 9 | left_wrist | 7 | LWrist |
| 10 | right_wrist | 4 | RWrist |
| 11 | left_hip | 12 | LHip |
| 12 | right_hip | 9 | RHip |
| 13 | left_knee | 13 | LKnee |
| 14 | right_knee | 10 | RKnee |
| 15 | left_ankle | 14 | LAnkle |
| 16 | right_ankle | 11 | RAnkle |

**8 BODY_25 joints are discarded** (no COCO equivalent): `1 Neck`, `8 MidHip`,
`19 LBigToe`, `20 LSmallToe`, `21 LHeel`, `22 RBigToe`, `23 RSmallToe`, `24 RHeel`.

The live mapping is served at `GET /api/keypoint-mapping` and rendered in the UI.

### Tensor contract

`mmaction.apis.inference_skeleton(model, pose_results, img_shape)` takes one dict
per frame:

```python
{'keypoints':       np.ndarray,  # (num_person, 17, 2)  RAW PIXELS
 'keypoint_scores': np.ndarray}  # (num_person, 17)
```

and builds the **M × T × V × C** tensor itself by running the config's own
`test_pipeline`:

```
PreNormalize2D → GenSkeFeat(j) → UniformSampleFrames(clip_len=100, num_clips=10)
→ PoseDecode → FormatGCNInput(num_person=2) → PackActionInputs
```

giving **M=2 persons, T=100 frames, V=17 joints, C=3 (x, y, score)**.

Coordinates are passed as **raw pixels** on purpose — `PreNormalize2D` applies
`(x − w/2)/(w/2)` using `img_shape`. Normalising beforehand would double-normalise
and silently corrupt the input.

Using the official API rather than hand-rolling the tensor is deliberate: shape,
padding and channel order stay owned by MMAction2.

### Documented limitations of the conversion

* The NTU RGB+D 2D skeletons this checkpoint trained on were produced by a
  **top-down HRNet** estimator, not OpenPose. Joint localisation and confidence
  distributions differ → domain shift even once the layout matches.
* Dropping Neck/MidHip removes BODY_25's torso-centre cues; COCO's graph centre
  is the nose.
* NTU RGB+D 60 has **no karate classes**.

---

## 10. Model and checkpoint

| | |
|---|---|
| Model | **ST-GCN** (`RecognizerGCN`, `STGCN` backbone, `GCNHead`) |
| Config | `stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py` |
| Checkpoint | `stgcn_..._20221129-484a394a.pth` (12.5 MB) |
| Dataset | **NTU RGB+D 60**, cross-subject split |
| Classes | 60 |
| Keypoint layout | `coco` → 17 nodes |
| Input | M×T×V×C = 2 × 100 × 17 × 3 |

The worker **refuses to run** if the keypoint count does not match the graph's
`num_node`, rather than producing meaningless scores.

---

## 11. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | readiness + full environment report |
| GET | `/api/keypoint-mapping` | the BODY_25→COCO-17 table |
| POST | `/api/analyze` | upload a video, start a job |
| GET | `/api/jobs` | list jobs |
| GET | `/api/jobs/{id}` | status + per-stage state |
| GET | `/api/jobs/{id}/result` | full result |
| GET | `/api/jobs/{id}/logs` | structured log entries |
| GET | `/api/jobs/{id}/skeleton` | summary; `?full=true` for all frames |
| GET | `/api/jobs/{id}/predictions` | MMAction2 output |
| GET | `/api/jobs/{id}/openpose-video` | H.264 overlay video |
| GET | `/api/jobs/{id}/input-video` | original upload |
| POST | `/api/live/start` · `/api/live/stop` | live session control |
| GET | `/api/live/status` | live snapshot (no image) |
| WS | `/api/live/ws` | live frames + predictions (~12 Hz) |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
     -F "file=@karate.mp4;type=video/mp4"
```

### Example response

```json
{ "job_id": "796a0aacd3c6", "status": "queued",
  "filename": "demo.mp4", "size_bytes": 635539 }
```

Then `GET /api/jobs/796a0aacd3c6` →

```json
{ "status": "running", "stage": "openpose",
  "stages": [ {"name": "upload", "state": "done", "duration_s": 0.26},
              {"name": "openpose", "state": "running"} ] }
```

---

## 12. Stage isolation and error handling

Each stage is tracked independently. A job ends as:

* `completed` — every stage succeeded
* `failed` — an early stage failed (validate / openpose / skeleton)
* **`partial`** — OpenPose and the skeleton succeeded but conversion or
  MMAction2 failed

`partial` is the important one: **an MMAction2 failure never invalidates the
OpenPose results**, which stay downloadable, and the UI says exactly that. Real
subprocess stderr is surfaced to the browser rather than a generic message.

Per job the backend logs: upload start/end, the exact OpenPose command and cwd,
OpenPose start/end, JSON frame count, people detected, tracking summary,
conversion start/end with slot counts, MMAction2 start/end, the prediction, and
any error.

---

## 13. Per-job artefacts

```
data/jobs/<job_id>/
├── input.mp4                    original upload (preserved)
├── input_trimmed.mp4            only if the frame cap applied
├── openpose/
│   ├── json/*_keypoints.json    RAW OpenPose output (never modified)
│   ├── output_video.avi         OpenPose MJPG overlay
│   └── output_video.mp4         H.264 transcode for the browser
├── skeleton/
│   ├── skeleton.json            parsed BODY_25 + track ids
│   └── skeleton_coco17.json     converted COCO-17 + conversion metadata
├── mmaction/prediction.json     top-1 / top-5 + provenance
├── result.json                  full job record
└── job.log                      per-job log
```

---

## 14. Real-time (live webcam) mode

```
webcam ── OpenPoseDemo.exe --camera (one long-lived process)
             │  writes one JSON + one JPEG per frame
             ▼
        watcher thread ── incremental centroid tracker ── BODY_25→COCO-17
             │                                              │
             │ rolling 100-frame buffer ────────────────────┘
             ▼
        persistent ST-GCN worker (model stays loaded)
             │
             ▼
        WebSocket /api/live/ws  ──►  browser
                                      left : clean camera frame
                                      right: BODY_25 skeleton on <canvas>
                                      below: live top-1 / top-5
```

### Why OpenPose owns the camera

The portable release only ships `pyopenpose.cp37-win_amd64.pyd`, a **CPython 3.7**
ABI extension that cannot be imported by Python 3.11/3.12 — and 3.7 is
end-of-life. So there is no in-process API available here, and
`OpenPoseDemo.exe --camera` is run once and left alive instead.

**The browser must not call `getUserMedia`.** This was tried and it broke the
pipeline: the webcam on this machine permits only one reader, the browser won a
race for it, and OpenPose then blocked **indefinitely** — alive, 184 MB
resident, **zero frames produced**, with the UI showing a perfect camera preview
next to an empty skeleton panel. Detection takes priority over preview, so the
left pane is fed by OpenPose's own frames.

Because the live pipeline runs with `REALTIME_RENDER_POSE=0`, those frames have
**no skeleton burnt in** — they are the clean camera image. The skeleton is drawn
client-side on a `<canvas>` from the streamed keypoints. That keeps the two panes
independent, avoids a doubled overlay, skips OpenPose's render step, and shrank
each frame from **343 KB to 190 KB**.

### Bandwidth and correctness notes

* At default webcam resolution each rendered JPEG was **343 KB** (~458 KB once
  base64-encoded); at 12 Hz that is ~5.5 MB/s down the socket and the view
  stutters. `REALTIME_CAMERA_RESOLUTION=640x480` plus `REALTIME_JPEG_EVERY`
  throttling keeps it usable. Keypoints and predictions still update every frame;
  only image transport is throttled.
* `img_shape` is **measured from the JPEG's SOF marker**, not taken from config.
  OpenPose reports keypoints in the camera's resolution and `PreNormalize2D`
  divides by `w/2` and `h/2`; a mismatched `img_shape` silently normalises every
  coordinate against the wrong extent and corrupts the prediction.
* Starting a session takes ~20 s (CUDA init + ST-GCN load). `POST /api/live/start`
  returns **409** if a start is already in progress, because a second click
  during that window would otherwise spawn a second OpenPose fighting for the
  camera.

---

## 15. Verification results

All figures below are measured on this machine, not estimated.

### Batch pipeline — `demo.mp4` (340×256, 280 frames)

| Stage | CPU build | **GPU + CUDA** |
|---|---|---|
| OpenPose BODY_25 | 120 frames in **262.5 s** — 2.187 s/frame (0.46 fps) | 280 frames in **19.8 s** — **0.071 s/frame (14.1 fps)** |
| Skeleton parse + tracking | < 0.1 s | < 0.1 s |
| MMAction2 one-shot (`num_clips=10`) | 4.87 s (CPU) | **1.18 s** (`cuda:0`) |
| MMAction2 model load | 0.85 s | 0.49 s |
| **End-to-end, 280 frames** | ~10 min | **~22 s** |

**OpenPose GPU vs CPU ≈ 30.8×.** The full pipeline now processes the entire
280-frame clip faster than the CPU build handled 10 frames.

> **Watch for orphaned processes.** An intermediate run measured only
> 0.782 s/frame. The cause was not the GPU: a stale `OpenPoseDemo.exe` orphaned
> by an earlier backend restart was still holding ~2 GB of the 4 GB VRAM
> (`nvidia-smi` showed 1955 MiB used, 59 % util, 85 °C). Killing it restored
> 0.071 s/frame — an **11× swing**. The backend now reaps such orphans at
> startup (`realtime_service.kill_orphaned_openpose`, scoped to the configured
> OpenPose directories) and stops live sessions on shutdown.

### Prediction consistency

| Run | Frames | `num_clips` | Top-1 | Score |
|---|---|---|---|---|
| CPU batch | 120 | 10 | walking towards each other | 0.7213 |
| GPU batch | 280 | 10 | walking towards each other | 0.5832 |
| Persistent worker | 120 | 1 | walking towards each other | 0.7320 |

The same top-1 across all three is the real signal that the conversion is
correct — a broken keypoint mapping produces incoherent labels. The score shifts
are explained: `num_clips` 10→1 removes temporal averaging, and the GPU run
covered 280 frames rather than 120.

Skeleton stats from the GPU run: 280 frames, 932 person detections, 3.33
people/frame, max 9 in one frame, 15 tracks; conversion filled 350 of 560 person
slots (210 zero-padded), `graph_num_node=17` matched `num_keypoints_sent=17`.

### Live mode

Measured **245 frames in 25 s ≈ 9.8 fps** on the GPU build, window filled
100/100, live inference **0.177 s**, one person tracked, clean frames 190 KB.

---

## 16. What worked, what failed, known limitations

**Worked**
* Full batch chain, browser → OpenPose → BODY_25 → COCO-17 → ST-GCN → browser.
* GPU OpenPose with **no CUDA toolkit and no cuDNN account** — the 1.7.0 GPU
  release already bundles `cublas64_11`, `cublasLt64_11`, `cudart64_110` and
  `curand64_10`, and it ran without the cuDNN sub-libraries, so the planned
  712 MB cuDNN download proved unnecessary.
* Real-time live webcam at ~10 fps with a rolling-window classifier.
* Stage isolation: an MMAction2 failure yields `partial`, never invalidating
  OpenPose output.

**Failed, and how it was resolved**
* **PyPI `mmaction2` 1.2.0 is broken** — missing `models/localizers/drn/`. Fixed
  by pointing the venv at the local checkout with a `.pth` file (writes nothing
  into the repo).
* **numpy 2.x breaks torch 2.1 + mmcv 2.1** (`_ARRAY_API not found`). Pinned
  `numpy<2`, installed last, `opencv-python==4.10.0.84`.
* **CMU's model server is offline** — BODY_25 weights come from a mirror with an
  asserted byte count.
* **`--frame_last` ≥ clip length aborts OpenPose.** The flag is now omitted
  unless it is genuinely needed and is clamped below the clip length.
* **Cross-volume `Move-Item` corrupted the GPU build.** File names and *sizes*
  all matched, but it died at DLL load with `0xC000012F` and printed nothing.
  The setup script now extracts in place and verifies by *running* the binary.
* **`getUserMedia` starved OpenPose of the camera** — see §14.
* mmengine prints to **stdout**, corrupting a bare JSON protocol; worker lines
  now carry an `@@MMA@@` prefix and the reader skips anything else.

**Known limitations**
* Predictions are **NTU RGB+D 60 labels**. Not karate. Out-of-domain.
* The tracker is nearest-centroid only: no re-identification, no occlusion
  handling; long gaps start a new track and can swap person slots.
* ST-GCN consumes a 100-frame window, so live output lags fast actions and
  cannot localise a technique to an instant.
* MMAction2 runs on CPU unless the CUDA venv is present; the gain there is ~0.1 s
  and OpenPose dominates runtime regardless.
* Single-worker, in-process job registry; jobs are lost on restart (artefacts on
  disk survive).
* The 4 GB VRAM budget constrains OpenPose `net_resolution`.

---

## 17. Next step: making it recognise karate

Nothing here has been trained or evaluated on karate, so the system **cannot**
classify Punch / Kick / Block. The pipeline is the deliverable; the classifier is
not. To get karate recognition:

1. **Collect and label a karate dataset.** Clips per technique (punch, kick,
   block, …), ideally ≥ 200–500 per class, varied performers, angles, lighting.
2. **Extract skeletons with this pipeline** — reuse `/api/analyze` to produce
   BODY_25 JSON, then the existing converter for COCO-17.
3. **Build an MMAction2 `PoseDataset` annotation pickle**: a list of dicts with
   `keypoint` `(M, T, V, 2)`, `keypoint_score` `(M, T, V)`, `frame_dir`, `label`,
   `img_shape`, `total_frames`; plus train/val splits.
4. **Fine-tune ST-GCN**: copy the NTU60 config, set `num_classes` to your class
   count, point `ann_file` at your pickle, and initialise from the NTU60
   checkpoint (`load_from`) so you keep its motion features.
5. **Evaluate honestly** — per-class accuracy and a confusion matrix on a
   *held-out performer*, not a random split, or you will leak identity.
6. Swap `CHECKPOINT_PATH`, `LABEL_MAP_PATH` and `MODEL_DATASET` in `.env`. The
   out-of-domain banner is driven by `MODEL_DATASET`, so it updates itself.

Only after step 5 shows real held-out accuracy should anyone describe this system
as recognising karate.
