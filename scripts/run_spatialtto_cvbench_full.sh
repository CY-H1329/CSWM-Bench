#!/bin/bash
# SpatialTTO on CV-Bench full dataset (~2638 samples)
# Train (TTO) on full + Eval on full, or inference-only with pre-trained score map.
#
# Usage (H100):
#   bash scripts/run_spatialtto_cvbench_full.sh                    # Train full + Eval full
#   bash scripts/run_spatialtto_cvbench_full.sh --max_steps 500     # Train 500 + Eval full
#   bash scripts/run_spatialtto_cvbench_full.sh --inference_only --score_map_path results/.../score_map_after_200.json
#   bash scripts/run_spatialtto_cvbench_full.sh --low_memory        # 3 agents (OOM fix)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "SpatialTTO: CV-Bench FULL (~2638 samples)"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python run_confidence_mas_step4_train_then_eval_frozen.py \
  --benchmark cvbench_full \
  --eval \
  --checkpoint_every 100 \
  "$@"
