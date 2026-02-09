#!/bin/bash
# H100에서 단일 → 멀티(3 agents 다수결) 순차 실행 + 정확도·틀린문제 비교 (한 프로그램)
# 사용법:
#   cd ~/CY/Spatial_MAS && bash scripts/run_h100_multiagent.sh
#   bash scripts/run_h100_multiagent.sh --max_per_category 50   # 빠른 테스트
#
# Conda 환경: spatial_mas (다르면 CONDA_ENV=... 로 지정)

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CONDA_ENV="${CONDA_ENV:-spatial_mas}"

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
  echo "[Spatial_MAS] Loaded .env"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "[Spatial_MAS] Unified: single-agent → multi-agent (3 agents, majority) + comparison"
echo "[Spatial_MAS] Running: python run_eval_unified.py --models qwen llava --split train --max_per_category 100 $*"
python run_eval_unified.py --models qwen llava --split train --max_per_category 100 "$@"

LATEST_RUN=$(ls -td results/*_unified 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
  echo "[Spatial_MAS] Done. Results: $LATEST_RUN"
  echo "[Spatial_MAS] Conversations: $LATEST_RUN/conversations/"
  if [ -f "$LATEST_RUN/comparison_single_vs_multi.txt" ]; then
    echo "[Spatial_MAS] Accuracy comparison:"
    cat "$LATEST_RUN/comparison_single_vs_multi.txt"
  fi
  if [ -f "$LATEST_RUN/wrong_comparison.txt" ]; then
    echo "[Spatial_MAS] Wrong-problem comparison:"
    cat "$LATEST_RUN/wrong_comparison.txt"
  fi
fi
