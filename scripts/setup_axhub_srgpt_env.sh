#!/usr/bin/env bash
# axhub.vaiv.kr (JupyterHub)에서 SpatialRGPT 전문가로 CV-Bench Count 평가용 환경 설정
#
# 사용 (순서 중요):
#   1) 가상환경 생성: conda create -n srgpt_axhub python=3.10 -y
#   2) 가상환경 활성화: conda activate srgpt_axhub
#   3) 환경설정 (pip install): cd /path/to/Spatial_MAS && bash scripts/setup_axhub_srgpt_env.sh
#   4) 실행: ./scripts/run_axhub_srgpt_cvbench_count.sh
#
# CUDA 버전: nvidia-smi로 확인. 기본 cu121. cu118 필요 시:
#   CUDA_VER=cu118 bash scripts/setup_axhub_srgpt_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRGPT_DIR="$PROJECT_ROOT/SpatialRGPT"
cd "$PROJECT_ROOT"

CUDA_VER="${CUDA_VER:-cu121}"

echo "=============================================="
echo "axhub SpatialRGPT 환경설정 (pip install)"
echo "  Project: $PROJECT_ROOT"
echo "  SpatialRGPT: $SRGPT_DIR"
echo "  PyTorch CUDA: $CUDA_VER"
echo "  (가상환경 활성화 후 실행하세요)"
echo "=============================================="

if [[ ! -d "$SRGPT_DIR" ]]; then
  echo "Error: SpatialRGPT/ not found. Clone it first:"
  echo "  git clone https://github.com/AnjieCheng/SpatialRGPT SpatialRGPT"
  exit 1
fi

pip install --upgrade pip

# PyTorch
echo ""
echo "[1] PyTorch ($CUDA_VER)..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

# FlashAttention
echo ""
echo "[2] FlashAttention..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash_attn 실패 → 폴백 사용"

# SpatialRGPT 의존성
echo ""
echo "[3] SpatialRGPT 의존성..."
pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales.git"
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.9.12
pip install sentencepiece shortuuid "pydantic<2" markdown2 requests httpx
pip install "accelerate>=0.27" peft "numpy<2" scikit-learn
pip install opencv-python pillow datasets openai
pip install "transformers>=4.51.0"

# SpatialRGPT
echo ""
echo "[4] SpatialRGPT (editable)..."
cd "$SRGPT_DIR"
pip install -e . --no-deps
cd "$PROJECT_ROOT"

# Spatial_MAS eval
pip install huggingface_hub qwen-vl-utils tqdm pyyaml

# flash_attn 폴백
echo ""
echo "[5] flash_attn 확인..."
if ! python -c "import flash_attn" 2>/dev/null; then
  echo "  flash_attn 없음 → 폴백 패치 적용"
  PATCH_SRC="$PROJECT_ROOT/patches/spatialrgpt_flash_attn_fallback.py"
  SRGPT_FLASH="$SRGPT_DIR/llava/model/multimodal_encoder/intern/flash_attention.py"
  if [[ -f "$PATCH_SRC" && -f "$SRGPT_FLASH" ]]; then
    cp "$PATCH_SRC" "$SRGPT_FLASH"
    echo "  폴백 적용 완료"
  fi
fi

echo ""
echo "=============================================="
echo "완료!"
echo ""
echo "다음 명령으로 테스트:"
echo "  cd $PROJECT_ROOT"
echo "  export SPATIALRGPT_PATH=$SRGPT_DIR"
echo "  ./scripts/run_axhub_srgpt_cvbench_count.sh"
echo ""
echo "또는 빠른 확인 (10개):"
echo "  python test_fixed_specialist_mas_v2.py --specialist spatial_rgpt --benchmark cvbench --category_filter Count --max_samples 10 --device cuda"
echo "=============================================="
