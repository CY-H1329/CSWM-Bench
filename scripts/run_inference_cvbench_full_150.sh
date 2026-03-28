#!/bin/bash
# Inference on CV-Bench full (~2638) with 150-step score map
# Run after: bash scripts/run_spatialtto_cvbench_150.sh (train 150)
#
# Usage:
#   bash scripts/run_inference_cvbench_full_150.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SCORE_MAP="${1:-results/spatialtto_150_frozen_cvbench/score_map_after_150.json}"

echo "=============================================="
echo "SpatialTTO Inference: CV-Bench FULL (~2638)"
echo "  Score map: $SCORE_MAP"
echo "=============================================="

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark cvbench_150 \
  --inference_only \
  --eval \
  --eval_full \
  --score_map_path "$SCORE_MAP" \
  "$@"
