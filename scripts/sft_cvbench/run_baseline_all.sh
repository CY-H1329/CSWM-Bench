#!/bin/bash
# 모든 모델 baseline (학습 없이, 600개 human_test)
# Usage: bash scripts/sft_cvbench/run_baseline_all.sh
set -e
cd "$(dirname "$0")/../.."

echo "=============================================="
echo "CV-Bench Baseline (zero-shot, 600 samples)"
echo "=============================================="

if [ ! -f data/sft_cvbench/splits/human_test.json ]; then
  python scripts/sft_cvbench/01_sample_dataset.py
fi

for MODEL in qwen3_4b llava4d sa2va spatialreasoner spatialrgpt; do
  echo ""
  echo "[Baseline] $MODEL..."
  python scripts/sft_cvbench/03_evaluate.py --model $MODEL --shots 0 --checkpoint base || true
done

echo ""
echo "[Aggregate] results..."
python scripts/sft_cvbench/04_aggregate_results.py

echo ""
echo "=============================================="
echo "Done. CSV: results/sft_cvbench/results_cvbench_scaling.csv"
echo "=============================================="
