#!/bin/bash
# Full SFT CV-Bench pipeline: sample → train → eval → aggregate
# Usage: bash scripts/sft_cvbench/run_full_pipeline.sh [shots]
set -e
cd "$(dirname "$0")/../.."
SHOTS=${1:-10}

echo "=============================================="
echo "SFT CV-Bench Pipeline (shots=$SHOTS)"
echo "=============================================="

# 1. Sample dataset
python scripts/sft_cvbench/01_sample_dataset.py

# 2. Train
for MODEL in qwen3_4b llava4d sa2va spatialreasoner; do
  echo ""
  echo "[2] Training $MODEL..."
  python scripts/sft_cvbench/02_train.py --model $MODEL --shots $SHOTS || true
done

# 3. Evaluate
for MODEL in qwen3_4b llava4d sa2va spatialreasoner; do
  CKPT="results/sft_cvbench/checkpoints/${MODEL}_cvbench_${SHOTS}shot"
  if [ -d "$CKPT" ]; then
    echo ""
    echo "[3] Evaluating $MODEL..."
    python scripts/sft_cvbench/03_evaluate.py --model $MODEL --shots $SHOTS --checkpoint $CKPT || true
  fi
done

# 4. Aggregate
echo ""
echo "[4] Aggregating results..."
python scripts/sft_cvbench/04_aggregate_results.py

echo ""
echo "=============================================="
echo "Done. See results/sft_cvbench/results_cvbench_scaling.csv"
echo "=============================================="
