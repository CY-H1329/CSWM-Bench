#!/bin/bash
# Inference on 3DSRBench FULL dataset (HuggingFace) with 150-step score map
# Run after: bash scripts/run_spatialtto_3dsrbench_150.sh (train 150)
#
# Usage:
#   bash scripts/run_inference_3dsrbench_full_150.sh   # Eval on full 3DSRBench (HuggingFace)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SCORE_MAP="results/spatialtto_150_frozen_3dsrbench/score_map_after_150.json"

echo "=============================================="
echo "SpatialTTO Inference: 3DSRBench FULL dataset"
echo "  Score map: $SCORE_MAP"
echo "=============================================="

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark 3dsrbench_150 \
  --inference_only \
  --eval \
  --eval_full \
  --score_map_path "$SCORE_MAP" \
  --no_spatial_rgpt \
  "$@"
