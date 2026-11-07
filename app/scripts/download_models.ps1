# Downloads every model weight the pipeline needs.
#
# WHY THIS EXISTS INSTEAD OF models/getBaseModels.bat
# ---------------------------------------------------
# OpenPose's bundled getBaseModels.bat pulls from
#   http://posefs1.perception.cs.cmu.edu/OpenPose/models/...
# which is OFFLINE (connection times out). The BODY_25 weights are therefore
# fetched from a public mirror. The expected size is asserted below; two
# independent mirrors report the identical byte count, which is the integrity
# check used here.

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$OpModelDir = Join-Path $Root "models\openpose\pose\body_25"
$MmModelDir = Join-Path $Root "models\mmaction"

New-Item -ItemType Directory -Force $OpModelDir | Out-Null
New-Item -ItemType Directory -Force $MmModelDir | Out-Null

function Get-IfMissing {
    param([string]$Url, [string]$Dest, [long]$ExpectedBytes = 0)
    if (Test-Path $Dest) {
        $len = (Get-Item $Dest).Length
        if ($ExpectedBytes -eq 0 -or $len -eq $ExpectedBytes) {
            Write-Host "  already present: $(Split-Path -Leaf $Dest) ($([math]::Round($len/1MB,1)) MB)"
            return
        }
        Write-Host "  size mismatch, re-downloading $(Split-Path -Leaf $Dest)"
    }
    Write-Host "  downloading $(Split-Path -Leaf $Dest) ..."
    curl.exe -sL --retry 3 -o $Dest $Url
    $len = (Get-Item $Dest).Length
    if ($ExpectedBytes -ne 0 -and $len -ne $ExpectedBytes) {
        throw "Downloaded $Dest is $len bytes, expected $ExpectedBytes"
    }
    Write-Host "  ok ($([math]::Round($len/1MB,1)) MB)"
}

Write-Host "==> OpenPose BODY_25 weights"
Get-IfMissing `
    -Url "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel" `
    -Dest (Join-Path $OpModelDir "pose_iter_584000.caffemodel") `
    -ExpectedBytes 104715850

# The prototxt ships with the OpenPose source checkout; copy it (read-only use,
# the original repo is never modified).
$Proto = Join-Path $OpModelDir "pose_deploy.prototxt"
if (-not (Test-Path $Proto)) {
    $SrcProto = Join-Path (Split-Path -Parent $Root) "openpose-master\models\pose\body_25\pose_deploy.prototxt"
    if (Test-Path $SrcProto) {
        Copy-Item $SrcProto $Proto
        Write-Host "  copied pose_deploy.prototxt from the OpenPose checkout"
    } else {
        throw "pose_deploy.prototxt not found at $SrcProto"
    }
}

Write-Host "==> MMAction2 ST-GCN checkpoint (NTU RGB+D 60, xsub, 2D keypoints)"
Get-IfMissing `
    -Url "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth" `
    -Dest (Join-Path $MmModelDir "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth") `
    -ExpectedBytes 12479895

Write-Host "==> NTU RGB+D 60 label map"
$LabelMap = Join-Path $MmModelDir "label_map_ntu60.txt"
if (-not (Test-Path $LabelMap)) {
    $SrcLabels = Join-Path (Split-Path -Parent $Root) "mmaction2-main\tools\data\skeleton\label_map_ntu60.txt"
    Copy-Item $SrcLabels $LabelMap
    Write-Host "  copied label_map_ntu60.txt from the MMAction2 checkout"
}

Write-Host ""
Write-Host "All model files ready."
