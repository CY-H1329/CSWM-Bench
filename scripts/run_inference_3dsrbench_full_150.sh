#!/bin/bash
# Inference on 3DSRBench with 150-step score map
# Run after: bash scripts/run_spatialtto_3dsrbench_150.sh (train 150)
#
# Usage:
#   bash scripts/run_inference_3dsrbench_full_150.sh              # Eval on 3dsrbench_500 (500)
#   bash scripts/run_inference_3dsrbench_full_150.sh --eval_full  # Eval on full HuggingFace
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SCORE_MAP="results/spatialtto_150_frozen_3dsrbench/score_map_after_150.json"

echo "=============================================="
echo "SpatialTTO Inference: 3DSRBench"
echo "  Score map: $SCORE_MAP"
echo "=============================================="

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark 3dsrbench_150 \
  --inference_only \
  --eval \
  --score_map_path "$SCORE_MAP" \
  --no_spatial_rgpt \
  "$@"
