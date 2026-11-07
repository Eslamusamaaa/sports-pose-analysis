# Fetches everything the repository deliberately does NOT contain.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File app/scripts/download_external.ps1
#
#   mmaction2-main/        MMAction2 source (read-only dependency)         -> <repo>/mmaction2-main
#   openpose-master/       OpenPose source (only the prototxt is needed)   -> <repo>/openpose-master
#   external/pose-models/  RTMDet-m + RTMPose-m (Halpe26) + YOLO11m-pose weights
#   external/pose-libs/    mmpose + mmdet (pip --target, so the MMAction2 venv is not touched)
#   external/yolo-libs/    ultralytics (pip --target)
#
# Not downloaded: OSNet re-ID weights (osnet_ain_x1_0_msmt17.pth). Re-ID is OPTIONAL;
# get the file from the torchreid model zoo (github.com/KaiyangZhou/deep-person-reid)
# and put it in app/backend/reid/ to enable appearance-based player re-identification.
param([switch]$SkipPoseLibs, [switch]$SkipYoloLibs)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Ext  = Join-Path $Repo "external"
New-Item -ItemType Directory -Force -Path (Join-Path $Ext "pose-models") | Out-Null

function Get-File($Url, $Dest) {
    if (Test-Path $Dest) { Write-Host "have  $(Split-Path -Leaf $Dest)"; return }
    Write-Host "get   $(Split-Path -Leaf $Dest)"
    curl.exe -L --fail --retry 5 -C - -o $Dest $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
}

# --- source checkouts (read-only dependencies) --------------------------------
if (-not (Test-Path (Join-Path $Repo "mmaction2-main"))) {
    git clone --depth 1 https://github.com/open-mmlab/mmaction2 (Join-Path $Repo "mmaction2-main")
}
if (-not (Test-Path (Join-Path $Repo "openpose-master"))) {
    git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose (Join-Path $Repo "openpose-master")
}

# --- pose-engine weights -------------------------------------------------------
$M = Join-Path $Ext "pose-models"
Get-File "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth" (Join-Path $M "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth")
Get-File "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" (Join-Path $M "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
Get-File "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt" (Join-Path $M "yolo11m-pose.pt")

# --- python libs, installed beside (not into) the MMAction2 venv ---------------
$VPy = Join-Path $Repo "app\third_party\mmaction-venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { $VPy = "python"; Write-Warning "MMAction2 venv not found, using system python (run app/scripts/setup_mmaction_env.ps1 first)" }

if (-not $SkipPoseLibs) {
    $Libs = Join-Path $Ext "pose-libs"
    & $VPy -m pip install --target $Libs --no-deps mmpose==1.3.2 mmdet==3.2.0 xtcocotools json_tricks munkres terminaltables shapely pycocotools tqdm
}
if (-not $SkipYoloLibs) {
    $Y = Join-Path $Ext "yolo-libs"
    & $VPy -m pip install --target $Y ultralytics lap
}
Write-Host "External dependencies ready under $Ext"
