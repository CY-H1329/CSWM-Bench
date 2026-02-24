#!/bin/bash
# Run explicit_3d_representation specialist test: 50 samples each from CV-Bench and 3DSRBench.
# Usage: bash scripts/run_explicit_3d_50.sh

set -e
cd "$(dirname "$0")/.."
echo "[Spatial_MAS] Running explicit_3d test: CV-Bench 50 + 3DSRBench 50"
python test_specialist_explicit_3d.py --benchmark both --max_samples 50 --show_failures 3
