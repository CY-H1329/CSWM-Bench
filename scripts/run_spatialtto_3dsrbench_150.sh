#!/bin/bash
# SpatialTTO: 3DSRBench 150-step optimization → fixed combination + score → inference
#
# 1. Train (TTO) 150 samples → score_map_after_150.json
# 2. Eval on frozen 3dsrbench_500 (500 samples)
#
# Usage (H100):
#   bash scripts/run_spatialtto_3dsrbench_150.sh              # Train 150 + Eval
#   bash scripts/run_spatialtto_3dsrbench_150.sh --inference_only  # Eval only (needs score map)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "SpatialTTO: 3DSRBench 150-step optimization"
echo "  → results/spatialtto_150_frozen_3dsrbench/"
echo "  → score_map_after_150.json"
echo "=============================================="

if [[ ! -d "data/dataset/3dsrbench_train_300" ]] && [[ " ${*:-} " != *" --inference_only "* ]]; then
  echo "WARNING: data/dataset/3dsrbench_train_300 not found."
  echo "  Run: python scripts/prepare_train_datasets.py --benchmarks 3dsrbench"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark 3dsrbench_150 \
  --eval \
  --checkpoint_every 50 \
  "$@"
