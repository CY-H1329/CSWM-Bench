#!/bin/bash
# MAS v2 + TTO (Trust Score) — H100 서버, 3DSRBench from data/dataset
#
# Multi-agent system + TTO (trust_score.py) update
# Specialists: llava4d, qwen3_4b, sa2va, spatial_rgpt (4 agents)
# Dataset: data/dataset/3dsrbench_train_300 (GitHub pull 후)
#
# Usage:
#   bash scripts/run_h100_mas_tto_3dsrbench.sh
#   bash scripts/run_h100_mas_tto_3dsrbench.sh --test_only --max_samples 20
#   bash scripts/run_h100_mas_tto_3dsrbench.sh --max_samples 100
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 4 specialists: no SpatialRGPT (requires SPATIALRGPT_PATH); use spatial_reasoner instead
WHITELIST_4="llava4d,qwen3_4b,sa2va,spatial_reasoner"

echo "=============================================="
echo "MAS v2 + TTO — H100 — 3DSRBench"
echo "Dataset: data/dataset/3dsrbench_train_300"
echo "Specialists: $WHITELIST_4"
echo "TTO: run_step4 (Beta + EMA)"
echo "=============================================="

# data/dataset 확인
if [[ ! -d "data/dataset/3dsrbench_train_300" ]]; then
  echo "ERROR: data/dataset/3dsrbench_train_300 not found."
  echo "  Run: git pull  (or clone from GitHub)"
  echo "  Or:  python scripts/prepare_train_datasets.py"
  exit 1
fi

python run_eval_mas_v2.py \
  --benchmark 3dsrbench \
  --dataset_subdir 3dsrbench_train_300 \
  --use_tto \
  --trust_step 4 \
  --use_vlm_reasoning \
  --specialist_whitelist "$WHITELIST_4" \
  --specialist_offload_after_use \
  --train_ratio 0.5 \
  --output_dir results/mas_v2_tto_3dsrbench \
  "$@"
