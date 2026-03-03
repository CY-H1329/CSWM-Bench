#!/bin/bash
# axhub.vaiv.kr에서 SpatialRGPT 전문가로 CV-Bench Count 평가
# Head + Final은 그대로, Specialist 3개 전부 SpatialRGPT
#
# 사용:
#   1) 환경 설정: bash scripts/setup_axhub_srgpt_env.sh
#   2) conda activate srgpt_axhub
#   3) ./scripts/run_axhub_srgpt_cvbench_count.sh
#
# 샘플 수 변경: MAX_SAMPLES=50 ./scripts/run_axhub_srgpt_cvbench_count.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export SPATIALRGPT_PATH="${SPATIALRGPT_PATH:-$PROJECT_ROOT/SpatialRGPT}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

if [[ ! -d "$SPATIALRGPT_PATH" ]]; then
  echo "Error: SpatialRGPT not found at $SPATIALRGPT_PATH"
  echo "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
  exit 1
fi

echo "=============================================="
echo "SpatialRGPT — CV-Bench Count ($MAX_SAMPLES samples)"
echo "  Head: Qwen3-VL-4B"
echo "  Specialist: SpatialRGPT (3 roles)"
echo "  Final: Qwen3-VL-8B"
echo "=============================================="

python test_fixed_specialist_mas_v2.py \
  --specialist spatial_rgpt \
  --benchmark cvbench \
  --category_filter Count \
  --max_samples "$MAX_SAMPLES" \
  --device cuda
