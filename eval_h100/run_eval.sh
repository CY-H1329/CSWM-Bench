#!/bin/bash
# Run SpatialLLM eval on 3DSRBench (H100)
# Usage: cd Spatial_MAS && bash eval_h100/run_eval.sh [max_samples]

set -e
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate spatial_mas 2>/dev/null || true

MAX_SAMPLES="${1:-50}"
echo "[eval_h100] SpatialReasoner on 3DSRBench, max_samples=$MAX_SAMPLES"

python evals_spatialllm/run_spatialllm_3dsrbench.py \
  --model_id ccvl/SpatialReasoner \
  --max_samples "$MAX_SAMPLES" \
  --seed 42 \
  --device cuda

echo ""
echo "[eval_h100] Done. Results in results/evals_spatialllm/"
