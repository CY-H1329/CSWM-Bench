#!/bin/bash
# SpatialTTO 3DSRBench: 3dsrbench_train_300 → TTO 고정 → frozen 3dsrbench_500 inference
#
# Usage:
#   bash scripts/run_spatialtto_3dsrbench.sh              # Train 300 + Eval frozen
#   bash scripts/run_spatialtto_3dsrbench.sh --inference_only   # Eval only
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/run_spatialtto_300_frozen.sh --benchmark 3dsrbench "$@"
