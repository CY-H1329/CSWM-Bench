#!/bin/bash
# H100에서 Qwen 단일 / LLaVA 단일 / Qwen+LLaVA 협력(2 agents) 비교
# 사용법:
#   cd ~/CY/Spatial_MAS && bash scripts/run_h100_collab.sh
#   bash scripts/run_h100_collab.sh --max_per_category 7
#   bash scripts/run_h100_collab.sh --tie_break llava

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CONDA_ENV="${CONDA_ENV:-spatialeval_orchestration}"

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
  echo "[Spatial_MAS] Loaded .env"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "[Spatial_MAS] Collab: Qwen only, LLaVA only, Qwen+LLaVA (2 agents)"
echo "[Spatial_MAS] Running: python run_eval_collab.py --split train --max_per_category 7 $*"
python run_eval_collab.py --split train --max_per_category 7 "$@"

LATEST_RUN=$(ls -td results/*_collab 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
  echo "[Spatial_MAS] Done. Results: $LATEST_RUN"
  if [ -f "$LATEST_RUN/comparison_collab.txt" ]; then
    echo "[Spatial_MAS] Comparison:"
    cat "$LATEST_RUN/comparison_collab.txt"
  fi
fi
