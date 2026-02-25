#!/bin/bash
# MAS v2 Baseline — H100 batch run
# Benchmarks: 3DSRBench, CV-Bench
# Sample sizes: 10, 50, 100
#
# Usage:
#   cd ~/CY/Spatial_MAS   # or your project path
#   bash experiments/mas_v2_baseline/run_h100.sh
#
# Uses --use_local_reasoning (DeepSeek-R1-Distill-Qwen-7B locally, no API)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_BASE="results/mas_v2_baseline"
SEED=42

echo "=============================================="
echo "MAS v2 Baseline — H100"
echo "Project: $PROJECT_ROOT"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

for BENCHMARK in cvbench 3dsrbench; do
  for N in 10 50 100; do
    echo ""
    echo ">>> $BENCHMARK | $N samples"
    echo "----------------------------------------------"
    python run_eval_mas_v2.py \
      --benchmark "$BENCHMARK" \
      --max_samples "$N" \
      --seed "$SEED" \
      --output_dir "$OUTPUT_BASE" \
      --use_local_reasoning \
      --device cuda
  done
done

echo ""
echo "=============================================="
echo "Done. Results in $OUTPUT_BASE/"
echo "=============================================="
