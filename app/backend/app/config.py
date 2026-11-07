"""Environment-driven configuration.

Every external path is resolved from a .env file so no absolute path is baked
into the source. See .env.example for the full list.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> backend/app -> backend -> karate-ai-web
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
PARENT_DIR = PROJECT_ROOT.parent  # holds openpose-master / mmaction2-main

load_dotenv(PROJECT_ROOT / ".env")


def _path(env_key: str, default: Path) -> Path:
    raw = os.getenv(env_key)
    if not raw or not raw.strip():
        return default
    p = Path(raw.strip().strip('"'))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(env_key: str, default: int) -> int:
    raw = os.getenv(env_key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# interpreter path -> (timestamp, probe result); see Settings._probe_torch_cuda.
_TORCH_PROBE_CACHE: dict[str, tuple[float, dict]] = {}
_TORCH_PROBE_TTL_S = 300.0

# Written by the setup scripts as their final step.
VENV_READY_MARKER = ".karate-venv-ready"


def _venv_is_ready(py: Path) -> bool:
    """True only if `py` belongs to a COMPLETELY provisioned venv.

    Presence of torch is not sufficient: the setup scripts install torch first
    and re-pin `numpy<2` last, and in between the venv imports torch but blows
    up with "Failed to initialize NumPy: _ARRAY_API not found". Switching to it
    during that window would break the pipeline, so readiness is signalled by an
    explicit marker file written only after verification succeeds.

    File-based rather than launching the interpreter: this is consulted on every
    health check and must stay cheap.
    """
    try:
        return py.exists() and (py.parent.parent / VENV_READY_MARKER).exists()
    except OSError:
        return False


@dataclass
class Settings:
    # --- OpenPose -----------------------------------------------------------
    openpose_dir: Path = field(
        default_factory=lambda: _path("OPENPOSE_DIR", PROJECT_ROOT / "third_party" / "openpose")
    )
    openpose_bin: Path = field(
        default_factory=lambda: _path(
            "OPENPOSE_BIN", PROJECT_ROOT / "third_party" / "openpose" / "bin" / "OpenPoseDemo.exe"
        )
    )
    openpose_model_dir: Path = field(
        default_factory=lambda: _path("OPENPOSE_MODEL_DIR", PROJECT_ROOT / "models" / "openpose")
    )
    openpose_net_resolution: str = field(
        default_factory=lambda: os.getenv("OPENPOSE_NET_RESOLUTION", "-1x256")
    )
    # 0 = no limit. Two caps because the two builds differ by ~20-30x in speed:
    # the CPU build runs ~2.2 s/frame (measured), the GPU build ~0.05-0.1 s/frame.
    # A cap that is comfortable on GPU would mean a multi-hour CPU run, so the
    # active build selects which one applies (see effective_max_frames).
    openpose_max_frames: int = field(default_factory=lambda: _int("OPENPOSE_MAX_FRAMES", 600))
    # Sliding-window analysis.
    # 30 frames = 1 second at 30fps. Shorter than the config's clip_len of 100
    # on purpose: a karate technique lasts ~0.3s, so a 3.3s window buries it in
    # seconds of footwork. UniformSampleFrames resamples the short window up to
    # 100 frames internally - verified working down to 15 frames.
    timeline_window: int = field(default_factory=lambda: _int("TIMELINE_WINDOW", 30))
    timeline_stride: int = field(default_factory=lambda: _int("TIMELINE_STRIDE", 15))
    timeline_enabled: bool = field(default_factory=lambda: _bool("TIMELINE_ENABLED", True))
    # Comma-separated model keys for the ensemble. Empty = the vetted default
    # (see model_zoo.DEFAULT_ENSEMBLE for the measurements behind the choice).
    ensemble_models: str = field(default_factory=lambda: os.getenv("ENSEMBLE_MODELS", ""))
    ensemble_enabled: bool = field(default_factory=lambda: _bool("ENSEMBLE_ENABLED", True))
    # "centroid" or "bytetrack". Default stays centroid - see bytetrack.py for
    # the measured comparison and why the result is NOT a clean win.
    tracker_backend: str = field(default_factory=lambda: os.getenv("TRACKER_BACKEND", "centroid"))
    openpose_max_frames_gpu: int = field(
        default_factory=lambda: _int("OPENPOSE_MAX_FRAMES_GPU", 1800)
    )
    openpose_timeout_s: int = field(default_factory=lambda: _int("OPENPOSE_TIMEOUT_S", 3600))
    openpose_render_pose: int = field(default_factory=lambda: _int("OPENPOSE_RENDER_POSE", 1))

    # GPU OpenPose build. When present it is preferred over the CPU build for
    # every stage; the CPU build stays as an automatic fallback.
    openpose_gpu_dir: Path = field(
        default_factory=lambda: _path(
            "OPENPOSE_GPU_DIR", PROJECT_ROOT / "third_party" / "openpose-gpu"
        )
    )
    openpose_prefer_gpu: bool = field(
        default_factory=lambda: _bool("OPENPOSE_PREFER_GPU", True)
    )

    # --- Real-time (live webcam) -------------------------------------------
    realtime_camera_index: int = field(default_factory=lambda: _int("REALTIME_CAMERA_INDEX", 0))
    realtime_net_resolution: str = field(
        default_factory=lambda: os.getenv("REALTIME_NET_RESOLUTION", "-1x192")
    )
    # Sliding window fed to ST-GCN. 30 frames at the live rate of ~14 fps is
    # about 2s of history. The old value of 100 meant the prediction described
    # the last ~7 SECONDS, so a punch was 1/7th of the window and the label
    # lagged far behind what the user had just done. Measured on batch footage,
    # a 1s window detected 6x more strikes than a 3.3s one.
    realtime_window: int = field(default_factory=lambda: _int("REALTIME_WINDOW", 30))
    # Karate live: the window is measured in SECONDS, not frames. 30 frames at
    # the ~20 fps live rate covered only ~1.5 s, which UniformSampleFrames then
    # stretched to 100 frames - every movement reached NTU60 about 2x slower
    # than in its training clips (which span a whole ~2-4 s action).
    realtime_window_s: float = field(
        default_factory=lambda: float(os.getenv("REALTIME_WINDOW_S", "2.5"))
    )
    # Run inference every N frames. At 0.014s per inference this is cheap, so
    # 5 frames (~3 predictions/second) is comfortably affordable.
    realtime_infer_every: int = field(default_factory=lambda: _int("REALTIME_INFER_EVERY", 5))
    # num_clips=1 trades temporal averaging for ~10x lower latency.
    realtime_num_clips: int = field(default_factory=lambda: _int("REALTIME_NUM_CLIPS", 1))
    realtime_max_people: int = field(default_factory=lambda: _int("REALTIME_MAX_PEOPLE", 4))
    # Capture size. A default-resolution webcam produced ~343 KB JPEGs, which
    # base64 to ~458 KB; at 12 Hz that is ~5.5 MB/s down the WebSocket and the
    # live view stutters. 640x480 keeps frames small and the coordinate space
    # predictable.
    realtime_camera_resolution: str = field(
        default_factory=lambda: os.getenv("REALTIME_CAMERA_RESOLUTION", "640x480")
    )
    # Ship a rendered frame every Nth processed frame. Keypoints and predictions
    # still update every frame; only the image transport is throttled.
    realtime_jpeg_every: int = field(default_factory=lambda: _int("REALTIME_JPEG_EVERY", 2))
    # 0 = write CLEAN frames (no skeleton burnt in). The live UI shows the
    # original on the left and draws the skeleton on a canvas on the right,
    # so burning the overlay into the frame would duplicate it - and skipping
    # the render is faster and yields smaller JPEGs.
    realtime_render_pose: int = field(default_factory=lambda: _int("REALTIME_RENDER_POSE", 0))

    # --- MMAction2 ----------------------------------------------------------
    mmaction2_dir: Path = field(
        default_factory=lambda: _path("MMACTION2_DIR", PARENT_DIR / "mmaction2-main")
    )
    mmaction_python: Path = field(
        default_factory=lambda: _path(
            "MMACTION_PYTHON",
            PROJECT_ROOT / "third_party" / "mmaction-venv" / "Scripts" / "python.exe",
        )
    )
    mmaction_config: Path = field(
        default_factory=lambda: _path(
            "MMACTION_CONFIG",
            PARENT_DIR
            / "mmaction2-main"
            / "configs"
            / "skeleton"
            / "stgcn"
            / "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
        )
    )
    checkpoint_path: Path = field(
        default_factory=lambda: _path(
            "CHECKPOINT_PATH",
            PROJECT_ROOT
            / "models"
            / "mmaction"
            / "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
        )
    )
    label_map_path: Path = field(
        default_factory=lambda: _path(
            "LABEL_MAP_PATH", PROJECT_ROOT / "models" / "mmaction" / "label_map_ntu60.txt"
        )
    )
    mmaction_timeout_s: int = field(default_factory=lambda: _int("MMACTION_TIMEOUT_S", 900))

    # Descriptive metadata surfaced in the UI so a prediction is never shown
    # without the provenance of the checkpoint that produced it.
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "ST-GCN"))
    model_dataset: str = field(default_factory=lambda: os.getenv("MODEL_DATASET", "NTU RGB+D 60"))
    model_keypoint_layout: str = field(default_factory=lambda: os.getenv("MODEL_LAYOUT", "coco"))

    # --- Swimming (separate page, separate fine-tuned ST-GCN) ---------------
    # Trained on SwimXYZ ground-truth BODY_25 joints resampled to 30 fps, in
    # contiguous 60-frame windows - so the timeline must send exactly that.
    swim_config: Path = field(
        default_factory=lambda: _path(
            "SWIM_CONFIG", PROJECT_ROOT / "models" / "swim" / "stgcn_swim4.py"
        )
    )
    swim_checkpoint: Path = field(
        default_factory=lambda: _path(
            "SWIM_CHECKPOINT", PROJECT_ROOT / "models" / "swim" / "stgcn_swim4.pth"
        )
    )
    swim_label_map: Path = field(
        default_factory=lambda: _path(
            "SWIM_LABEL_MAP", PROJECT_ROOT / "models" / "swim" / "label_map_swim4.txt"
        )
    )
    swim_window: int = field(default_factory=lambda: _int("SWIM_WINDOW", 60))
    swim_stride: int = field(default_factory=lambda: _int("SWIM_STRIDE", 30))
    # Windows whose top score is below this are reported as "uncertain" rather
    # than forced into one of the four strokes (the model has no "other" class).
    swim_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("SWIM_MIN_CONFIDENCE", "0.6"))
    )

    # --- Alternative pose engines (both sports) -----------------------------
    # Installed OUTSIDE the MMAction2 venv's site-packages (pip --target) so the
    # venv that MMAction2 depends on is never modified; pose_extract.py puts
    # these folders on sys.path itself.
    pose_libs_dir: Path = field(
        default_factory=lambda: _path("POSE_LIBS_DIR", Path(r"D:/Users/karate-training/swim/pose-libs"))
    )
    yolo_libs_dir: Path = field(
        default_factory=lambda: _path("YOLO_LIBS_DIR", Path(r"D:/Users/karate-training/yolo-libs"))
    )
    pose_models_dir: Path = field(
        default_factory=lambda: _path("POSE_MODELS_DIR", Path(r"D:/Users/karate-training/swim/pose-models"))
    )

    # Person re-identification (OSNet-AIN x1.0, MSMT17; MIT). Keeps "Player 1"
    # on the same fighter by appearance when the tracker swaps or loses ids.
    reid_dir: Path = field(
        default_factory=lambda: _path("REID_DIR", Path(r"D:/Users/karate-training/reid"))
    )

    @property
    def reid_cfg(self) -> dict | None:
        weights = self.reid_dir / "osnet_ain_x1_0_msmt17.pth"
        if not (weights.exists() and (self.reid_dir / "osnet_ain.py").exists()):
            return None
        return {"lib": str(self.reid_dir), "weights": str(weights)}

    def pose_engine_models(self, engine: str) -> dict:
        libs, m = self.pose_libs_dir, self.pose_models_dir
        if engine == "rtmpose":
            return {
                "det_config": str(libs / "mmpose/.mim/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"),
                "det_weights": str(m / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
                "pose_config": str(libs / "mmpose/.mim/configs/body_2d_keypoint/rtmpose/body8/"
                                          "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"),
                "pose_weights": str(m / "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"),
            }
        if engine == "yolo11":
            return {"yolo_weights": str(m / "yolo11m-pose.pt")}
        return {}

    def pose_engine_ready(self, engine: str) -> bool:
        if engine == "openpose":
            return True
        files = self.pose_engine_models(engine)
        lib = self.pose_libs_dir if engine == "rtmpose" else self.yolo_libs_dir
        return bool(files) and lib.exists() and all(Path(p).exists() for p in files.values())

    @property
    def swim_model_ready(self) -> bool:
        return all(p.exists() for p in (self.swim_config, self.swim_checkpoint, self.swim_label_map))

    # --- Runtime ------------------------------------------------------------
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda:0"))
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", PROJECT_ROOT / "data"))
    ffmpeg_bin: str = field(default_factory=lambda: os.getenv("FFMPEG_BIN", "ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: os.getenv("FFPROBE_BIN", "ffprobe"))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 512))
    cors_origins: str = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:5173")
    )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def body25_model_file(self) -> Path:
        return self.openpose_model_dir / "pose" / "body_25" / "pose_iter_584000.caffemodel"

    # --- which OpenPose build is actually active ----------------------------
    @property
    def cpu_mmaction_python(self) -> Path:
        return PROJECT_ROOT / "third_party" / "mmaction-venv" / "Scripts" / "python.exe"

    @property
    def active_mmaction_python(self) -> Path:
        """Configured interpreter, falling back to the CPU venv if it is absent.

        MMACTION_PYTHON may point at the CUDA venv on D: while that venv is
        still being built. Without this fallback the MMAction2 stage would fail
        outright during the transition instead of simply running on CPU.

        Note `python.exe` appears as soon as `python -m venv` runs, long before
        torch finishes installing, so presence of the interpreter is NOT enough -
        we require torch to actually be importable from that venv.
        """
        if _venv_is_ready(self.mmaction_python):
            return self.mmaction_python
        if _venv_is_ready(self.cpu_mmaction_python):
            return self.cpu_mmaction_python
        # Neither is complete; return the configured one so the error names it.
        return self.mmaction_python

    @property
    def gpu_openpose_bin(self) -> Path:
        return self.openpose_gpu_dir / "bin" / "OpenPoseDemo.exe"

    @property
    def gpu_openpose_available(self) -> bool:
        return self.openpose_prefer_gpu and self.gpu_openpose_bin.exists()

    @property
    def active_openpose_bin(self) -> Path:
        """GPU build when installed, otherwise the CPU build."""
        return self.gpu_openpose_bin if self.gpu_openpose_available else self.openpose_bin

    @property
    def active_openpose_dir(self) -> Path:
        return self.openpose_gpu_dir if self.gpu_openpose_available else self.openpose_dir

    @property
    def openpose_device_label(self) -> str:
        return "cuda (GPU build)" if self.gpu_openpose_available else "cpu (CPU-only build)"

    @property
    def effective_max_frames(self) -> int:
        """Frame cap for the build that will actually run. 0 = unlimited."""
        return (
            self.openpose_max_frames_gpu
            if self.gpu_openpose_available
            else self.openpose_max_frames
        )

    def estimate_runtime_s(self, frames: int) -> float:
        """Rough wall-clock estimate, so the UI can warn before a long run."""
        per_frame = 0.08 if self.gpu_openpose_available else 2.2
        return round(frames * per_frame, 1)

    # Real-time uses the same resolution rules as the batch path.
    @property
    def realtime_openpose_bin(self) -> Path:
        return self.active_openpose_bin

    @property
    def realtime_device_label(self) -> str:
        return self.openpose_device_label

    @property
    def realtime_frame_hw(self) -> tuple[int, int]:
        """Frame size reported to PreNormalize2D for live inference.

        OpenPose renders the camera at its native size; 480p is the common
        default for USB webcams. Override if the camera differs, since a wrong
        img_shape skews the coordinate normalisation.
        """
        raw = os.getenv("REALTIME_FRAME_HW", "480x640")
        try:
            h, w = raw.lower().split("x")
            return int(h), int(w)
        except ValueError:
            return 480, 640

    # --- Health -------------------------------------------------------------
    def resolved_device(self) -> str:
        """Report the device that will actually be used, never just the request.

        OpenPose here is a CPU-only prebuilt binary; this reports the device for
        the MMAction2 stage, which is the part that can honour DEVICE.
        """
        if self.device.startswith("cpu"):
            return "cpu"
        probe = self._probe_torch_cuda()
        return self.device if probe.get("cuda_available") else "cpu"

    def _probe_torch_cuda(self) -> dict:
        """Ask the MMAction2 venv whether torch sees CUDA.

        Launching that interpreter and importing torch takes ~8 s, and every
        /api/health call did it - so pages waited ~7 s for their first status,
        and a page opened during a backend restart kept its live button disabled.
        The answer only changes when the venv changes, so successes are cached.
        """
        if not self.active_mmaction_python.exists():
            return {"cuda_available": False, "reason": "mmaction venv missing"}
        key = str(self.active_mmaction_python)
        hit = _TORCH_PROBE_CACHE.get(key)
        if hit and time.time() - hit[0] < _TORCH_PROBE_TTL_S:
            return hit[1]
        result = self._probe_torch_cuda_uncached()
        if "torch" in result:  # never cache a failed probe
            _TORCH_PROBE_CACHE[key] = (time.time(), result)
        return result

    def _probe_torch_cuda_uncached(self) -> dict:
        try:
            out = subprocess.run(
                [
                    str(self.active_mmaction_python),
                    "-c",
                    "import json,torch;print(json.dumps({'cuda_available':torch.cuda.is_available(),"
                    "'torch':torch.__version__,'cuda':torch.version.cuda}))",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if out.returncode == 0:
                import json as _json

                return _json.loads(out.stdout.strip().splitlines()[-1])
            return {"cuda_available": False, "reason": out.stderr[-400:]}
        except Exception as exc:  # noqa: BLE001 - health probe must never raise
            return {"cuda_available": False, "reason": str(exc)}

    def environment_report(self) -> dict:
        """Everything the UI needs to show whether each stage can run."""
        torch_info = self._probe_torch_cuda()
        return {
            "openpose": {
                # Report the build that will ACTUALLY run, not the CPU default.
                "dir": str(self.active_openpose_dir),
                "binary": str(self.active_openpose_bin),
                "binary_found": self.active_openpose_bin.exists(),
                "cpu_build_dir": str(self.openpose_dir),
                "gpu_build_dir": str(self.openpose_gpu_dir),
                "seconds_per_frame_estimate": 0.08 if self.gpu_openpose_available else 2.2,
                "model_dir": str(self.openpose_model_dir),
                "body25_weights_found": self.body25_model_file.exists(),
                "net_resolution": self.openpose_net_resolution,
                "max_frames": self.effective_max_frames,
                "device": self.openpose_device_label,
                "gpu_build_available": self.gpu_openpose_available,
            },
            "mmaction": {
                "dir": str(self.mmaction2_dir),
                "python": str(self.active_mmaction_python),
                "python_found": self.active_mmaction_python.exists(),
                "python_configured": str(self.mmaction_python),
                "using_cpu_fallback_venv": self.active_mmaction_python != self.mmaction_python,
                "config": str(self.mmaction_config),
                "config_found": self.mmaction_config.exists(),
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_found": self.checkpoint_path.exists(),
                "label_map": str(self.label_map_path),
                "label_map_found": self.label_map_path.exists(),
                "model_name": self.model_name,
                "dataset": self.model_dataset,
                "layout": self.model_keypoint_layout,
                "torch": torch_info.get("torch"),
                "torch_cuda": torch_info.get("cuda"),
                "cuda_available": torch_info.get("cuda_available", False),
                "device_requested": self.device,
                "device_effective": self.resolved_device(),
            },
            "ffmpeg": {
                "ffmpeg": shutil.which(self.ffmpeg_bin) or self.ffmpeg_bin,
                "ffmpeg_found": shutil.which(self.ffmpeg_bin) is not None,
                "ffprobe_found": shutil.which(self.ffprobe_bin) is not None,
            },
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
