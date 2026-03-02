#!/bin/bash
# MAS + TTO — 5 specialists (SpatialRGPT 제외)
#
# 5 specialists: llava4d, qwen3_4b, sa2va, spaceom, spatial_reasoner
# Dataset: data/dataset/3dsrbench_train_300
# TTO: run_step4 (Beta + EMA)
#
# Usage:
#   bash scripts/run_mas_tto.sh
#   bash scripts/run_mas_tto.sh --test_only --max_samples 20
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 5 agents: SpaceOm + SpatialReasoner, SpatialRGPT 제외
WHITELIST="llava4d,qwen3_4b,sa2va,spaceom,spatial_reasoner"

echo "=============================================="
echo "MAS + TTO"
echo "Specialists: $WHITELIST"
echo "Dataset: data/dataset/3dsrbench_train_300"
echo "=============================================="

if [[ ! -d "data/dataset/3dsrbench_train_300" ]]; then
  echo "ERROR: data/dataset/3dsrbench_train_300 not found."
  echo "  Run: git pull  or  python scripts/prepare_train_datasets.py"
  exit 1
fi

python run_eval_mas_v2.py \
  --benchmark 3dsrbench \
  --dataset_subdir 3dsrbench_train_300 \
  --use_tto \
  --trust_step 4 \
  --use_vlm_reasoning \
  --specialist_whitelist "$WHITELIST" \
  --specialist_offload_after_use \
  --train_ratio 0.5 \
  --output_dir results/mas_tto_3dsrbench \
  "$@"
