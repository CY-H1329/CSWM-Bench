#!/bin/bash
# SpatialRGPT 전문가로 Count 100 평가
# cvbench_400 Count 카테고리 100개
#
# 사용법:
#   1) 의존성 설치 (최초 1회): bash scripts/install_spatialrgpt_deps.sh
#   2) conda activate spatial_mas   # 또는 spatial_mas_rgpt
#   3) ./scripts/run_spatialrgpt_cvbench_count100.sh
#
# 또는 CONDA_ENV 지정:
#   CONDA_ENV=spatial_mas ./scripts/run_spatialrgpt_cvbench_count100.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# SpatialRGPT 경로 (프로젝트 내 SpatialRGPT/ 또는 환경변수)
if [[ -z "$SPATIALRGPT_PATH" ]]; then
  if [[ -d "$PROJECT_ROOT/SpatialRGPT" ]]; then
    export SPATIALRGPT_PATH="$PROJECT_ROOT/SpatialRGPT"
    echo "[SpatialRGPT] Using SPATIALRGPT_PATH=$SPATIALRGPT_PATH"
  else
    echo "Error: SPATIALRGPT_PATH not set and SpatialRGPT/ not found."
    echo "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
    exit 1
  fi
fi

# Conda env (선택)
if [[ -n "$CONDA_ENV" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  echo "[SpatialRGPT] Activated conda env: $CONDA_ENV"
fi

echo "=============================================="
echo "SpatialRGPT — Count 100 (cvbench)"
echo "=============================================="

python test_fixed_specialist_mas_v2.py \
  --specialist spatial_rgpt \
  --benchmark cvbench \
  --category_filter Count \
  --max_samples 100 \
  --device cuda
