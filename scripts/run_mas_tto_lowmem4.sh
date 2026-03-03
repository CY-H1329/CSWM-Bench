#!/bin/bash
# MAS + TTO — Low-memory 4-agent (SpaceOm inclus)
#
# Variante de run_mas_tto.sh pour tester 4-agent en mode low-memory:
#   qwen3_4b, llava4d, spaceom, spatial_reasoner
#
# Usage:
#   LOW_MEMORY=1 bash scripts/run_mas_tto_lowmem4.sh --test_only --max_samples 20
#   bash scripts/run_mas_tto_lowmem4.sh --low_memory --test_only --max_samples 20
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 4-agent low-memory: qwen3_4b, llava4d, spaceom, spatial_reasoner
EXTRA_ARGS=()
for a in "$@"; do
  if [[ "$a" == "--low_memory" ]]; then LOW_MEMORY=1; else EXTRA_ARGS+=("$a"); fi
done
if [[ -n "$LOW_MEMORY" ]]; then
  WHITELIST="qwen3_4b,llava4d,spaceom,spatial_reasoner"
  echo "[LOW_MEMORY] 4-agent mode (SpaceOm): $WHITELIST"
else
  WHITELIST="llava4d,qwen3_4b,sa2va,spaceom,spatial_reasoner"
fi

echo "=============================================="
echo "MAS + TTO (lowmem4)"
echo "Specialists: $WHITELIST"
echo "Dataset: data/dataset/3dsrbench_train_300"
echo "=============================================="

if [[ ! -d "data/dataset/3dsrbench_train_300" ]]; then
  echo "ERROR: data/dataset/3dsrbench_train_300 not found."
  echo "  Run: git pull  or  python scripts/prepare_train_datasets.py"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python run_eval_mas_v2.py \
  --benchmark 3dsrbench \
  --dataset_subdir 3dsrbench_train_300 \
  --use_tto \
  --trust_step 4 \
  --use_vlm_reasoning \
  --specialist_whitelist "$WHITELIST" \
  --specialist_offload_after_use \
  --train_ratio 0.5 \
  --output_dir results/mas_tto_lowmem4_3dsrbench \
  "${EXTRA_ARGS[@]}"
