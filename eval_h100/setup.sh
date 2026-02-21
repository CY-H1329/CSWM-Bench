#!/bin/bash
# H100 setup: install deps and verify env
# Usage: cd Spatial_MAS && bash eval_h100/setup.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[eval_h100] Project root: $ROOT"

# 1) Conda env
if conda env list 2>/dev/null | grep -q "spatial_mas "; then
  echo "[eval_h100] Conda env 'spatial_mas' exists."
  eval "$(conda shell.bash hook)"
  conda activate spatial_mas
else
  echo "[eval_h100] Creating conda env from environment.yml ..."
  conda env create -f environment.yml
  eval "$(conda shell.bash hook)"
  conda activate spatial_mas
fi

# 2) Pip deps
echo "[eval_h100] Installing pip deps ..."
pip install -r eval_h100/requirements_h100.txt -q

# 3) CUDA check
echo "[eval_h100] Checking CUDA ..."
python -c "
import torch
cuda = torch.cuda.is_available()
print('  CUDA available:', cuda)
if cuda:
    print('  Device:', torch.cuda.get_device_name(0))
"

# 4) Quick import test
echo "[eval_h100] Testing imports ..."
python -c "
import sys
sys.path.insert(0, '.')
from src.models import get_runner, list_agents
print('  Registry OK:', list(list_agents().keys())[:5], '...')
"

echo ""
echo "[eval_h100] Setup done. Run: bash eval_h100/test_runners.sh"
echo "[eval_h100] Or: bash eval_h100/run_eval.sh"
