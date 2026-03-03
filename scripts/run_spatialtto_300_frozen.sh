#!/bin/bash
# SpatialTTO: train 300 → TTO 고정 → frozen inference
# 4 agents: qwen3_4b, llava4d, spaceom, spatial_reasoner
#
# Usage:
#   bash scripts/run_spatialtto_300_frozen.sh --benchmark cvbench
#   bash scripts/run_spatialtto_300_frozen.sh --benchmark 3dsrbench
#   bash scripts/run_spatialtto_300_frozen.sh --benchmark 3dsrbench --inference_only
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BENCHMARK="cvbench"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--benchmark" ]]; then
    BENCHMARK="$2"
    shift 2
  else
    EXTRA_ARGS+=("$1")
    shift
  fi
done

TRAIN_DIR="data/dataset/${BENCHMARK}_train_300"
echo "=============================================="
echo "SpatialTTO: ${BENCHMARK}_train_300 → frozen ${BENCHMARK}"
echo "4 agents: qwen3_4b, llava4d, spaceom, spatial_reasoner"
echo "=============================================="

if [[ ! -d "$TRAIN_DIR" ]] && [[ " ${EXTRA_ARGS[*]} " != *" --inference_only "* ]]; then
  echo "ERROR: $TRAIN_DIR not found."
  echo "  Run: python scripts/prepare_train_datasets.py"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark "$BENCHMARK" "${EXTRA_ARGS[@]}"
