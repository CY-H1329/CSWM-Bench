#!/bin/bash
# H100 서버용 MAS + TTO 설정 스크립트
#
# 1. GitHub에서 Spatial_MAS pull (또는 clone)
# 2. trust_score (Spatial_AOMAS) 확인/복사
# 3. data/dataset 확인
#
# Usage:
#   bash scripts/setup_h100_mas_tto.sh
#   bash scripts/setup_h100_mas_tto.sh /path/to/workdir
#
set -e

WORKDIR="${1:-$(pwd)}"
REPO_URL="https://github.com/CY-H1329/Spatial_MAS.git"

echo "=============================================="
echo "MAS + TTO H100 Setup"
echo "Workdir: $WORKDIR"
echo "=============================================="

cd "$WORKDIR"

# 1. Spatial_MAS (이미 있으면 pull, 없으면 clone)
if [[ -d "Spatial_MAS" ]]; then
  echo "[1] Pulling Spatial_MAS..."
  cd Spatial_MAS
  git pull origin main || git pull origin master || true
  cd ..
else
  echo "[1] Cloning Spatial_MAS..."
  git clone "$REPO_URL"
  cd Spatial_MAS
fi

cd Spatial_MAS
PROJECT_ROOT="$(pwd)"

# 2. trust_score 확인 (spatial_aomas/trust_score.py)
if [[ -f "spatial_aomas/trust_score.py" ]]; then
  echo "[2] trust_score: spatial_aomas/trust_score.py OK"
elif [[ -f "../Spatial_AOMAS/trust_score.py" ]]; then
  echo "[2] Copying trust_score from Spatial_AOMAS..."
  mkdir -p spatial_aomas
  cp ../Spatial_AOMAS/trust_score.py spatial_aomas/
else
  echo "[2] WARNING: trust_score.py not found. TTO (--use_tto) will fallback to default updater."
  echo "    Add Spatial_AOMAS/trust_score.py or ensure spatial_aomas/trust_score.py exists."
fi

# 3. data/dataset 확인
if [[ -d "data/dataset/3dsrbench_train_300" ]]; then
  echo "[3] data/dataset/3dsrbench_train_300 OK"
else
  echo "[3] data/dataset/3dsrbench_train_300 not found."
  echo "    Run: python scripts/prepare_train_datasets.py"
  echo "    Or ensure git pull brought data/dataset."
fi

echo ""
echo "Setup done. Run:"
echo "  cd $PROJECT_ROOT"
echo "  bash scripts/run_h100_mas_tto_3dsrbench.sh"
echo ""
echo "Test (20 samples):"
echo "  bash scripts/run_h100_mas_tto_3dsrbench.sh --test_only --max_samples 20"
echo ""
