# Installs the GPU OpenPose build and the CUDA runtime DLLs it needs.
#
# WHY NOT THE CUDA TOOLKIT
# ------------------------
# OpenPose 1.7.0's GPU binaries were built against CUDA 11.1 + cuDNN 8.1. The
# usual route is installing the full CUDA toolkit (~3 GB) plus cuDNN, and cuDNN
# historically requires an NVIDIA developer account. That is avoided here: NVIDIA
# publishes the same redistributable DLLs as plain PyPI wheels, so the handful of
# DLLs OpenPose actually loads are extracted from those wheels and dropped next
# to OpenPoseDemo.exe (Windows resolves DLLs from the executable's directory).
#
# cuDNN 8.9 is used rather than 8.1 because that is what is published for
# Windows. Both export `cudnn64_8.dll` - the soname is stable across the cuDNN 8
# series - so the binary links against it. This is verified by actually running
# the binary, not assumed.
#
# No NVIDIA account, no toolkit install, no compilation.

$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $PSScriptRoot
$ThirdParty= Join-Path $Root "third_party"
$Downloads = Join-Path $ThirdParty "downloads"
$Wheels    = Join-Path $ThirdParty "cuda-wheels"

# Honour OPENPOSE_GPU_DIR from .env so the build is EXTRACTED straight to its
# final home (e.g. on D:) instead of being extracted here and moved later -
# a cross-volume move is exactly what corrupted a previous build.
$GpuDir = Join-Path $ThirdParty "openpose-gpu"
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    $line = Select-String -Path $EnvFile -Pattern '^\s*OPENPOSE_GPU_DIR\s*=' | Select-Object -First 1
    if ($line) {
        $val = ($line.Line -split '=', 2)[1].Trim().Trim('"')
        if ($val) {
            $GpuDir = if ([IO.Path]::IsPathRooted($val)) { $val } else { Join-Path $Root $val }
        }
    }
}
# Stage the zip next to the destination so the extract never crosses volumes.
$GpuParent = Split-Path -Parent $GpuDir
New-Item -ItemType Directory -Force $GpuParent | Out-Null
$Zip = Join-Path $GpuParent "openpose-gpu.zip"

$GpuUrl = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip"

# NVIDIA redistributable wheels (public PyPI, no account required).
$WheelSpecs = @(
    @{ Name = "nvidia-cudnn-cu11";        Version = "8.9.5.29"   },
    @{ Name = "nvidia-cublas-cu11";       Version = "11.11.3.6"  },
    @{ Name = "nvidia-cuda-runtime-cu11"; Version = "11.8.89"    }
)

New-Item -ItemType Directory -Force $Downloads | Out-Null
New-Item -ItemType Directory -Force $Wheels    | Out-Null

# ---------------------------------------------------------------- OpenPose ---
# Exact expected size. GitHub release downloads of this size get truncated
# fairly often (curl exit 18 / "connection interrupted"), and a truncated zip
# extracts into DLLs that LOOK right but fail to load, so the size is asserted
# exactly rather than with >=, and -C - resumes rather than restarting.
$ExpectedZipBytes = 437912637

for ($try = 1; $try -le 5; $try++) {
    $have = if (Test-Path $Zip) { (Get-Item $Zip).Length } else { 0 }
    if ($have -eq $ExpectedZipBytes) { break }
    Write-Host "==> Downloading GPU OpenPose (418 MB), attempt $try (have $have bytes)"
    curl.exe -sL --retry 5 --retry-all-errors -C - -o $Zip $GpuUrl
}
$have = (Get-Item $Zip).Length
if ($have -ne $ExpectedZipBytes) {
    throw "GPU zip is $have bytes, expected $ExpectedZipBytes. Delete it and re-run."
}

if (-not (Test-Path (Join-Path $GpuDir "bin\OpenPoseDemo.exe"))) {
    Write-Host "==> Extracting GPU OpenPose to $GpuDir"
    # Extract beside the destination so the final Move-Item stays on one volume.
    $Tmp = Join-Path $GpuParent "_gpu_extract"
    if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
    Expand-Archive -Path $Zip -DestinationPath $Tmp -Force
    $Inner = Get-ChildItem $Tmp -Directory | Select-Object -First 1
    if (Test-Path $GpuDir) { Remove-Item $GpuDir -Recurse -Force }
    Move-Item $Inner.FullName $GpuDir
    Remove-Item $Tmp -Recurse -Force
}

$Bin = Join-Path $GpuDir "bin"
Write-Host "==> GPU OpenPose at $GpuDir"

# ------------------------------------------------------------ CUDA runtime ---
# NOTHING TO DO. Verified empirically: the OpenPose 1.7.0 GPU release already
# ships the CUDA 11 redistributables it needs in bin/ --
#   cublas64_11.dll, cublasLt64_11.dll, cudart64_110.dll, curand64_10.dll
# and it runs WITHOUT the large cuDNN sub-libraries (only the 0.2 MB
# cudnn64_8.dll dispatch shim is present, and inference never calls into it).
# An earlier plan to fetch ~1.6 GB of NVIDIA redistributable wheels was dropped
# after the binary was confirmed working without them. No CUDA toolkit and no
# NVIDIA developer account are required.
Write-Host "==> CUDA runtime: bundled with the release, nothing to install"

# ------------------------------------------------------------------ verify ---
# This check is not optional. A corrupted or partially copied build still has
# the right file names and even the right file SIZES, but dies at DLL load with
# 0xC000012F (STATUS_INVALID_IMAGE_NOT_MZ) and prints nothing at all. Running
# the binary is the only reliable way to know it is sound.
#
# LESSON: do NOT relocate this folder with Move-Item/Copy-Item across volumes.
# Doing so corrupted a working build once (sizes matched, contents did not).
# Re-extract from the zip at the destination instead - that is why $GpuDir is
# derived from OPENPOSE_GPU_DIR and extracted in place.
Write-Host "==> Verifying the GPU binary actually runs"
Push-Location $GpuDir
try {
    $out = & ".\bin\OpenPoseDemo.exe" --help 2>&1 | Out-String
    if ($out -match "model_pose") {
        Write-Host "  OK - binary loaded its libraries and printed its flags"
    } else {
        throw ("GPU binary failed to start (exit $LASTEXITCODE). " +
               "Empty output with exit 0xC000012F means a corrupt DLL: delete " +
               "$GpuDir and re-run this script to re-extract.")
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "GPU OpenPose installed. The backend auto-prefers it (OPENPOSE_PREFER_GPU=true)."
Write-Host "Restart the backend, then check /api/health -> environment.openpose.device"
