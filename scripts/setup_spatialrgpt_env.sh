#!/bin/bash
# Spatial RGPT 전용 가상환경 생성 (새 터미널에서 실행)
# 사용법: ./scripts/setup_spatialrgpt_env.sh [full|minimal]
#   full (기본): SpatialRGPT 전체 의존성 설치 (VILA, flash_attn 등)
#   minimal: spatial_mas env만 사용, SPATIALRGPT_PATH만 설정 (가벼움)

set -e
cd "$(dirname "$0")/.."

MODE=${1:-full}
ENV_NAME="spatial_rgpt"
SPATIALRGPT_DIR="$(pwd)/SpatialRGPT"
PROJECT_ROOT="$(pwd)"

if [ ! -d "$SPATIALRGPT_DIR" ]; then
    echo "SpatialRGPT 폴더가 없습니다. 먼저 클론하세요:"
    echo "  git clone https://github.com/AnjieCheng/SpatialRGPT $SPATIALRGPT_DIR"
    exit 1
fi

if [ "$MODE" = "minimal" ]; then
    echo "=== Spatial RGPT (minimal) - spatial_mas env 사용 ==="
    echo "Spatial_MAS 테스트만 할 경우 이 방법을 권장합니다."
    echo ""
    echo "1. spatial_mas 환경 생성 (없으면):"
    echo "   conda env create -f environment.yml"
    echo ""
    echo "2. 사용 시:"
    echo "   conda activate spatial_mas"
    echo "   export SPATIALRGPT_PATH=$SPATIALRGPT_DIR"
    echo "   cd $PROJECT_ROOT"
    echo "   python test_fixed_specialist_mas_v2.py --specialist spatial_rgpt --benchmark cvbench --category_filter Count --max_samples 100"
    echo ""
    echo "참고: docs/SPATIALRGPT_SERVER_SETUP.md 에 패치 적용 방법이 있습니다."
    exit 0
fi

echo "=== Spatial RGPT 가상환경 (full) 설정 ==="
echo "환경 이름: $ENV_NAME"
echo "SpatialRGPT: $SPATIALRGPT_DIR"
echo ""

cd "$SPATIALRGPT_DIR"
./environment_setup.sh $ENV_NAME

echo ""
echo "=== 완료. 사용법 ==="
echo "  conda activate $ENV_NAME"
echo "  export SPATIALRGPT_PATH=$SPATIALRGPT_DIR"
echo "  cd $PROJECT_ROOT"
echo "  python test_fixed_specialist_mas_v2.py --specialist spatial_rgpt --benchmark cvbench --category_filter Count --max_samples 100"
echo ""
