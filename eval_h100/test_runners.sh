#!/bin/bash
# Run runner tests on H100
# Usage: cd Spatial_MAS && bash eval_h100/test_runners.sh

set -e
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate spatial_mas 2>/dev/null || true

echo "[eval_h100] Running test_runners.py ..."
python eval_h100/test_runners.py --model qwen3_4b --device cuda
