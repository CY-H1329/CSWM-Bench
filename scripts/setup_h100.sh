#!/bin/bash
# H100 환경 설정 (pull 후 한 번만 실행)
# 사용법: cd ~/CY/Spatial_MAS && bash scripts/setup_h100.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "[Spatial_MAS] Project root: $ROOT"

# 1) Conda 환경
if conda env list | grep -q "spatial_mas "; then
  echo "[Spatial_MAS] Conda env 'spatial_mas' already exists."
else
  echo "[Spatial_MAS] Creating conda env from environment.yml ..."
  conda env create -f environment.yml
fi

# 2) CUDA PyTorch (선택, conda pytorch가 GPU 안 잡을 때만)
echo "[Spatial_MAS] Checking PyTorch CUDA ..."
eval "$(conda shell.bash hook)"
conda activate spatial_mas
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[Spatial_MAS] PyTorch sees CUDA. Skip pip torch."
else
  echo "[Spatial_MAS] Installing PyTorch with CUDA 12.1 ..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# 3) API 키 안내
if [ -f "$ROOT/.env" ]; then
  echo "[Spatial_MAS] .env found. run_h100.sh will load it."
else
  echo ""
  echo "[Spatial_MAS] Create .env for API key (optional for GPT):"
  echo "  echo 'export OPENAI_API_KEY=sk-your-key' > $ROOT/.env"
  echo "  (or export OPENAI_API_KEY=... before running)"
fi
echo "[Spatial_MAS] Setup done. Run: bash scripts/run_h100.sh"
