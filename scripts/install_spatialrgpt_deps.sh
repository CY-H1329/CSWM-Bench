#!/bin/bash
# SpatialRGPT + Spatial_MAS 의존성 한 번에 설치
# 사용: bash scripts/install_spatialrgpt_deps.sh
#       또는: conda activate your_env && bash scripts/install_spatialrgpt_deps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "SpatialRGPT + Spatial_MAS 의존성 설치"
echo "=============================================="

# 1. 기본 의존성
echo ""
echo "[1/4] pip install -r requirements-spatialrgpt.txt"
pip install -r requirements-spatialrgpt.txt

# 2. flash-attn 시도 (CUDA 환경에서만 성공 가능)
echo ""
echo "[2/4] flash-attn 설치 시도 (실패해도 계속 진행)"
pip install flash-attn 2>/dev/null || echo "  flash-attn 설치 실패 → 폴백 패치 적용 예정"

# 3. flash_attn 사용 가능 여부 확인
echo ""
echo "[3/4] flash_attn import 확인"
FLASH_OK=0
python -c "import flash_attn" 2>/dev/null && FLASH_OK=1 || true
if [[ "$FLASH_OK" -eq 1 ]]; then
    echo "  flash_attn OK"
else
    echo "  flash_attn 없음 → 폴백 패치 적용"
fi

# 4. flash_attn 없으면 폴백 패치 적용
if [[ "$FLASH_OK" -ne 1 ]]; then
    echo ""
    echo "[4/4] FlashAttention 폴백 패치 적용"
    SRGPT_FLASH="$PROJECT_ROOT/SpatialRGPT/llava/model/multimodal_encoder/intern/flash_attention.py"
    PATCH_SRC="$PROJECT_ROOT/patches/spatialrgpt_flash_attn_fallback.py"
    if [[ -f "$SRGPT_FLASH" && -f "$PATCH_SRC" ]]; then
        cp "$PATCH_SRC" "$SRGPT_FLASH"
        echo "  $SRGPT_FLASH → 폴백으로 교체됨"
    else
        echo "  경고: 패치 파일 없음. flash-attn을 수동 설치하세요: pip install flash-attn"
    fi
else
    echo "[4/4] flash_attn 사용 가능, 패치 불필요"
fi

echo ""
echo "=============================================="
echo "설치 완료"
echo "=============================================="
echo "실행: ./scripts/run_spatialrgpt_cvbench_count100.sh"
