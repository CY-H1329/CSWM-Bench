#!/bin/bash
# SpatialReasoner baseline (학습 없이, 600개 human_test)
# Uses spatial prompt by default (paper-like; add --no_spatial_prompt for raw Q+options)
set -e
cd "$(dirname "$0")/../.."

echo "=============================================="
echo "SpatialReasoner Baseline (600 samples)"
echo "=============================================="

if [ ! -f data/sft_cvbench/splits/human_test.json ]; then
  python scripts/sft_cvbench/01_sample_dataset.py
fi

python scripts/sft_cvbench/03_evaluate.py --model spatialreasoner --shots 0 --checkpoint base

echo ""
echo "Done. Results: results/sft_cvbench/spatialreasoner/0/human_test/results.json"
