# Builds a CUDA-enabled MMAction2 venv on D:.
#
# WHY D:
#   The CUDA torch wheel is ~2.7 GB and expands to ~4 GB. C: has under 6 GB
#   free, so installing there would risk filling the system drive. D: has ample
#   space. pip's cache AND temp directory are both redirected to D: as well -
#   otherwise pip still stages the 2.7 GB download on C: and fills it anyway.
#
# The backend only needs MMACTION_PYTHON to point here; everything else is
# unchanged, because the MMAction2 stage already runs as a subprocess.

$ErrorActionPreference = "Stop"

$Py311   = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"

# D:\ root is not writable without elevation on this machine, so pick the first
# writable candidate instead of assuming the drive root.
function Find-WritableBase {
    foreach ($cand in @("D:\Users", "D:\Downloads", "D:\")) {
        if (-not (Test-Path $cand)) { continue }
        try {
            $probe = Join-Path $cand ("_wt_" + [guid]::NewGuid().ToString('N').Substring(0, 6) + ".tmp")
            [IO.File]::WriteAllText($probe, "x")
            Remove-Item $probe -Force
            return (Join-Path $cand "karate-ai-cuda")
        } catch { continue }
    }
    throw "No writable location found on D:. Create a folder there and pass it as CUDA_BASE."
}

$Base    = if ($env:CUDA_BASE) { $env:CUDA_BASE } else { Find-WritableBase }
Write-Host "==> CUDA base: $Base"
$VenvDir = Join-Path $Base "mmaction-venv"
$PipTmp  = Join-Path $Base "pip-tmp"
$PipCache= Join-Path $Base "pip-cache"

$Root = Split-Path -Parent $PSScriptRoot
$Mm2  = Join-Path (Split-Path -Parent $Root) "mmaction2-main"

if (-not (Test-Path $Py311)) { throw "Python 3.11 not found at $Py311" }
if (-not (Test-Path "D:\"))  { throw "D: drive not available" }

New-Item -ItemType Directory -Force $Base, $PipTmp, $PipCache | Out-Null

# Keep every large temporary file off C:.
$env:TMP  = $PipTmp
$env:TEMP = $PipTmp
$env:PIP_CACHE_DIR = $PipCache

Write-Host "==> Creating CUDA venv at $VenvDir"
if (Test-Path $VenvDir) { Remove-Item $VenvDir -Recurse -Force }
& $Py311 -m venv $VenvDir
$VPy = Join-Path $VenvDir "Scripts\python.exe"

& $VPy -m pip install --upgrade pip --quiet

Write-Host "==> Installing torch 2.1.0+cu118 (~2.7 GB, downloading to D:)"
& $VPy -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 `
    --index-url https://download.pytorch.org/whl/cu118

# Same numpy-1.x ABI constraint as the CPU venv (see setup_mmaction_env.ps1).
Write-Host "==> Installing mmengine + deps"
& $VPy -m pip install "mmengine>=0.7.1,<1.0.0" "opencv-python==4.10.0.84" `
    importlib_metadata scipy einops matplotlib "pillow<11" --quiet

Write-Host "==> Installing mmcv 2.1.0 (prebuilt cu118 wheel - no compilation)"
& $VPy -m pip install mmcv==2.1.0 `
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

Write-Host "==> Linking the LOCAL mmaction2 repo (repo stays untouched)"
$SitePackages = Join-Path $VenvDir "Lib\site-packages"
Set-Content -Path (Join-Path $SitePackages "_local_mmaction2.pth") -Value $Mm2 -Encoding ascii

Write-Host "==> Re-pinning numpy < 2 (must be last)"
& $VPy -m pip install "numpy<2" --quiet

Write-Host "==> Verifying CUDA"
& $VPy -c @"
import torch, mmcv, mmaction, numpy
print('torch     ', torch.__version__)
print('numpy     ', numpy.__version__)
print('mmcv      ', mmcv.__version__)
print('mmaction2 ', mmaction.__version__)
print('cuda_avail', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device    ', torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device='cuda') @ torch.randn(512, 512, device='cuda')
    torch.cuda.synchronize()
    print('matmul OK ', tuple(x.shape))
from mmaction.apis import init_recognizer, inference_skeleton
print('mmaction apis OK')
"@

Write-Host ""
Write-Host "Done. Point the backend at it by setting in .env:"
Write-Host "  MMACTION_PYTHON=$VPy"

# Signal completion ONLY now: the backend refuses to use a venv without this
# marker, so a half-installed environment is never selected.
New-Item -ItemType File -Force (Join-Path $VenvDir ".karate-venv-ready") | Out-Null
Write-Host "Marked venv ready."
