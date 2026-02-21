#!/bin/bash
# Run SpatialLLM/SpatialReasoner on 3DSRBench (100 samples) on H100
#
# Usage:
#   bash evals_spatialllm/run_h100.sh
#   bash evals_spatialllm/run_h100.sh 50    # 50 samples
#   bash evals_spatialllm/run_h100.sh 100 ccvl/SpatialReasoner-SFT

set -e
cd "$(dirname "$0")/.."

MAX_SAMPLES="${1:-100}"
MODEL_ID="${2:-ccvl/SpatialReasoner}"

echo "[SpatialLLM] 3DSRBench, max_samples=$MAX_SAMPLES, model=$MODEL_ID"

python evals_spatialllm/run_spatialllm_3dsrbench.py \
  --model_id "$MODEL_ID" \
  --max_samples "$MAX_SAMPLES" \
  --seed 42 \
  --device cuda

echo ""
echo "Done. Results in results/evals_spatialllm/"
