#!/bin/bash
# SFT CV-Bench pipeline for H100
# Run from project root: bash scripts/sft_cvbench/run_h100_sft.sh

set -e
cd "$(dirname "$0")/../.."

echo "[SFT] 1. Sampling dataset..."
python scripts/sft_cvbench/01_sample_dataset.py

echo "[SFT] 2. Training (example: qwen3_4b, 10 shot)..."
python scripts/sft_cvbench/02_train.py --model qwen3_4b --shots 10

echo "[SFT] 3. Evaluation (run after training completes)..."
# python scripts/sft_cvbench/03_evaluate.py --model qwen3_4b --shots 10 --checkpoint results/sft_cvbench/checkpoints/qwen3_4b_cvbench_10shot

echo "[SFT] 4. Aggregate results..."
python scripts/sft_cvbench/04_aggregate_results.py

echo "[SFT] Done."
