#!/bin/bash
# SpatialRGPT baseline (zero-shot) on human_test (600 samples).
# Requires: export SPATIALRGPT_PATH=/path/to/SpatialRGPT
# Or use run_baseline_all.sh for all models.
set -e
cd "$(dirname "$0")/../.."

if [ ! -f data/sft_cvbench/splits/human_test.json ]; then
  python scripts/sft_cvbench/01_sample_dataset.py
fi

python scripts/sft_cvbench/03_evaluate.py --model spatialrgpt --shots 0 --checkpoint base
python scripts/sft_cvbench/04_aggregate_results.py
echo "Done. CSV: results/sft_cvbench/results_cvbench_scaling.csv"
