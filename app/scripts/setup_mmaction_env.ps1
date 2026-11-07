# Creates the isolated Python 3.11 virtualenv used ONLY for MMAction2 inference.
#
# Why a separate venv:
#   mmcv has no prebuilt wheel for Python 3.12 / torch 2.13, and this machine has
#   no C++ compiler (no MSVC, no CMake), so mmcv CANNOT be built from source here.
#   The newest combination with a prebuilt Windows wheel is Python 3.11 + torch 2.1.0.
#   The FastAPI backend keeps running on the system Python 3.12 and talks to this
#   venv over a subprocess, which also isolates MMAction2 crashes from the API.
#
# numpy is pinned <2 because mmcv 2.1.0 / mmaction2 1.2.0 are built against the
# numpy 1.x C ABI and segfault or fail to import against numpy 2.x.

$ErrorActionPreference = "Stop"

$Py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$Root = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $Root "third_party\mmaction-venv"

if (-not (Test-Path $Py311)) { throw "Python 3.11 not found at $Py311" }

Write-Host "==> Creating venv at $VenvDir"
& $Py311 -m venv $VenvDir

$VPy = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "==> Upgrading pip"
& $VPy -m pip install --upgrade pip --quiet

Write-Host "==> Installing torch 2.1.0 (CPU build)"
& $VPy -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu --quiet

Write-Host "==> Pinning numpy < 2 (mmcv 2.1.0 is built against the numpy 1.x ABI)"
& $VPy -m pip install "numpy<2" --quiet

Write-Host "==> Installing mmengine"
& $VPy -m pip install "mmengine>=0.7.1,<1.0.0" --quiet

Write-Host "==> Installing mmcv 2.1.0 (prebuilt CPU wheel, no compilation)"
& $VPy -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html --quiet

# opencv-python is pinned because 5.x hard-requires numpy>=2, which is
# incompatible with the mmcv/torch numpy-1.x ABI pin below.
# importlib_metadata is imported by mmaction/utils/dependency.py but is not
# declared by the package, so it must be installed explicitly.
Write-Host "==> Installing mmaction2 runtime deps"
& $VPy -m pip install "opencv-python==4.10.0.84" importlib_metadata scipy einops matplotlib "pillow<11" --quiet

# The PyPI mmaction2 1.2.0 wheel is BROKEN: it omits
# mmaction/models/localizers/drn/, so `import mmaction.models` dies with
#   ModuleNotFoundError: No module named 'mmaction.models.localizers.drn'
# The local mmaction2-main checkout does contain it, so we point the venv at the
# local repo with a .pth file. This is read-only: unlike `pip install -e .` it
# writes nothing into the repo (no .egg-info, no build artefacts).
Write-Host "==> Linking the LOCAL mmaction2 repo via .pth (repo stays untouched)"
$Mm2 = Join-Path (Split-Path -Parent $Root) "mmaction2-main"
if (-not (Test-Path (Join-Path $Mm2 "mmaction\__init__.py"))) {
    throw "Local mmaction2 repo not found at $Mm2"
}
$SitePackages = Join-Path $VenvDir "Lib\site-packages"
Set-Content -Path (Join-Path $SitePackages "_local_mmaction2.pth") -Value $Mm2 -Encoding ascii
& $VPy -m pip uninstall mmaction2 -y --quiet 2>$null

# numpy LAST and pinned: opencv/matplotlib pull numpy 2.x, which breaks both
# torch 2.1.0's NumPy bridge ("Failed to initialize NumPy: _ARRAY_API not
# found") and mmcv 2.1.0, as both are compiled against the numpy 1.x C ABI.
Write-Host "==> Re-pinning numpy < 2 (must be last; opencv/matplotlib pull numpy 2.x)"
& $VPy -m pip install "numpy<2" --quiet

Write-Host "==> Verifying"
& $VPy -c @"
import torch, mmengine, mmcv, mmaction, numpy
print('torch      ', torch.__version__, 'cuda:', torch.cuda.is_available())
print('numpy      ', numpy.__version__)
print('mmengine   ', mmengine.__version__)
print('mmcv       ', mmcv.__version__)
print('mmaction2  ', mmaction.__version__)
from mmaction.models import STGCN
print('STGCN import OK')
"@

Write-Host "==> MMAction2 venv ready: $VPy"

# Signal completion ONLY now: the backend refuses to use a venv without this
# marker, so a half-installed environment is never selected.
New-Item -ItemType File -Force (Join-Path $VenvDir ".karate-venv-ready") | Out-Null
Write-Host "Marked venv ready."
