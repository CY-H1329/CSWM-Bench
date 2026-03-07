#!/bin/bash
# SpatialTTO: 150-step optimization → fixed combination + score → inference
#
# 1. Train (TTO) 150 samples → score_map_after_150.json
# 2. Eval on frozen cvbench_400 with fixed combination
#
# Usage (H100):
#   bash scripts/run_spatialtto_cvbench_150.sh              # Train 150 + Eval (no SpatialRGPT)
#   bash scripts/run_spatialtto_cvbench_150.sh --inference_only  # Eval only (needs score map)
#
# Inference with 150-step combination (other scripts):
#   python run_inference_fixed_spatialtto.py --score_map_path results/spatialtto_150_frozen_cvbench/score_map_after_150.json --benchmark cvbench --max_samples 400
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "SpatialTTO: CV-Bench 150-step optimization"
echo "  → results/spatialtto_150_frozen_cvbench/"
echo "  → score_map_after_150.json"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark cvbench_150 \
  --eval \
  --no_spatial_rgpt \
  --checkpoint_every 50 \
  "$@"
