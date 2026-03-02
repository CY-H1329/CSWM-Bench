#!/usr/bin/env bash
# SpatialRGPT 전용 환경 설정 (Spatial_MAS에서 SpatialRGPT specialist 사용 가능)
#
# 사용:
#   # 기존 환경 삭제 (선택)
#   conda env remove -n srgpt -y
#
#   cd SpatialRGPT
#   bash setup_env_srgpt.sh
#   conda activate srgpt
#
# 또는 프로젝트 루트에서:
#   bash SpatialRGPT/setup_env_srgpt.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"

ENV_NAME="${1:-srgpt}"
echo "=============================================="
echo "SpatialRGPT 환경 생성: $ENV_NAME"
echo "  SpatialRGPT: $SCRIPT_DIR"
echo "  Project: $PROJECT_ROOT"
echo "=============================================="

# 0. 기존 env 있으면 제거 (깨끗한 설치)
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[0] 기존 환경 '$ENV_NAME' 제거..."
  conda env remove -n "$ENV_NAME" -y
fi

# 1. Conda env 생성
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# 2. pip 업그레이드
pip install --upgrade pip

# 3. PyTorch (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 4. FlashAttention 시도
echo ""
echo "[4] FlashAttention 설치..."
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "  flash_attn OK"
else
    echo "  flash_attn 실패 → 폴백 패치 사용"
fi

# 5. SpatialRGPT 의존성 (pyproject 기반, transformers는 나중에)
pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales.git"
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.9.12
pip install sentencepiece shortuuid "pydantic<2" markdown2 requests httpx
pip install "accelerate>=0.27" peft "numpy<2" scikit-learn
pip install opencv-python pillow datasets openai

# 6. transformers 4.51 (Qwen3 + no_init_weights 패치로 SpatialRGPT 호환)
pip install "transformers>=4.51.0"

# 7. SpatialRGPT (editable)
pip install -e . --no-deps

# 8. Spatial_MAS eval
pip install huggingface_hub qwen-vl-utils tqdm pyyaml

# 9. flash_attn 없으면 폴백 패치
echo ""
echo "[9] flash_attn 최종 확인..."
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "  flash_attn 없음 → 폴백 패치 적용"
    PATCH_SRC="$PROJECT_ROOT/patches/spatialrgpt_flash_attn_fallback.py"
    SRGPT_FLASH="$SCRIPT_DIR/llava/model/multimodal_encoder/intern/flash_attention.py"
    if [[ -f "$PATCH_SRC" && -f "$SRGPT_FLASH" ]]; then
        cp "$PATCH_SRC" "$SRGPT_FLASH"
        echo "  폴백 적용 완료"
    else
        echo "  경고: 패치 파일 없음 ($PATCH_SRC)"
    fi
fi

echo ""
echo "=============================================="
echo "완료!"
echo "  conda activate $ENV_NAME"
echo "  cd $PROJECT_ROOT"
echo "  export SPATIALRGPT_PATH=$SCRIPT_DIR"
echo "  ./scripts/run_spatialrgpt_cvbench_count100.sh"
echo "=============================================="
