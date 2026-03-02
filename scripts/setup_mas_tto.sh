#!/bin/bash
# MAS + TTO 실행용 폴더 생성
#
# 5 specialists: llava4d, qwen3_4b, sa2va, spaceom, spatial_reasoner
# SpatialRGPT 제외
#
# Usage:
#   cd Spatial_MAS && bash scripts/setup_mas_tto.sh
#   bash scripts/setup_mas_tto.sh /path/to/parent
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_MAS="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT="${1:-$(dirname "$CURRENT_MAS")}"
NEW_DIR="$PARENT/Spatial_MAS_TTO"
REPO_URL="https://github.com/CY-H1329/Spatial_MAS.git"

echo "=============================================="
echo "MAS + TTO 폴더 생성"
echo "Target: $NEW_DIR"
echo "=============================================="

if [[ -d "$NEW_DIR" ]]; then
  echo "이미 존재합니다. 최신화..."
  if [[ -f "$CURRENT_MAS/src2/models/spaceom.py" ]]; then
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      "$CURRENT_MAS/" "$NEW_DIR/" 2>/dev/null || cp -r "$CURRENT_MAS"/* "$NEW_DIR/" 2>/dev/null || true
  else
    cd "$NEW_DIR" && git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
  fi
else
  if [[ -f "$CURRENT_MAS/src2/models/spaceom.py" ]]; then
    echo "현재 Spatial_MAS에서 복사..."
    cp -r "$CURRENT_MAS" "$NEW_DIR"
    rm -rf "$NEW_DIR/.git" 2>/dev/null || true
  else
    echo "Clone 중..."
    git clone "$REPO_URL" "$NEW_DIR"
  fi
  cd "$NEW_DIR"
fi

cd "$NEW_DIR"
echo ""
echo "설치 완료. 실행:"
echo "  cd $NEW_DIR"
echo "  bash scripts/run_mas_tto.sh"
echo ""
echo "테스트 (20 samples):"
echo "  bash scripts/run_mas_tto.sh --test_only --max_samples 20"
echo ""
