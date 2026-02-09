#!/bin/bash
# H100에서 Multi-agent 평가 (3 agents, 다수결) + 단일 vs 다중 비교
# 사용법:
#   cd ~/CY/Spatial_MAS && bash scripts/run_h100_multiagent.sh
#   bash scripts/run_h100_multiagent.sh --baseline_run_dir results/20260209_175745
#   bash scripts/run_h100_multiagent.sh --max_per_category 50   # 빠른 테스트
#
# Conda 환경: spatial_mas (다르면 아래 CONDA_ENV 수정)

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CONDA_ENV="${CONDA_ENV:-spatial_mas}"
# 서버에서 다른 환경 쓰면: CONDA_ENV=spatialeval_orchestration bash scripts/run_h100_multiagent.sh

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
  echo "[Spatial_MAS] Loaded .env"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "[Spatial_MAS] Multi-agent (3 agents, majority vote)"
echo "[Spatial_MAS] Running: python run_eval_multiagent.py --models qwen llava --split train --max_per_category 100 $*"
python run_eval_multiagent.py --models qwen llava --split train --max_per_category 100 "$@"

LATEST_RUN=$(ls -td results/*_multiagent 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
  echo "[Spatial_MAS] Done. Results: $LATEST_RUN"
  echo "[Spatial_MAS] Conversations: $LATEST_RUN/conversations/"
  if [ -f "$LATEST_RUN/comparison_single_vs_multi.txt" ]; then
    echo "[Spatial_MAS] Comparison (single vs multi):"
    cat "$LATEST_RUN/comparison_single_vs_multi.txt"
  fi
fi
