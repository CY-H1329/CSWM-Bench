#!/bin/bash
# H100에서 평가 + 실패 분석 + 틀린 샘플 저장 (pull 후 실행)
# 사용법: cd ~/CY/Spatial_MAS && bash scripts/run_h100.sh
# 옵션:   bash scripts/run_h100.sh --max_samples 100   (빠른 테스트)

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# .env 있으면 로드 (OPENAI_API_KEY 등)
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
  echo "[Spatial_MAS] Loaded .env"
fi

# Conda 활성화 (서버 환경: spatialeval_orchestration / 로컬: CONDA_ENV=spatial_mas 등으로 변경 가능)
CONDA_ENV="${CONDA_ENV:-spatialeval_orchestration}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# GPU 지정 (필요 시 여기서 수정)
# export CUDA_VISIBLE_DEVICES=0

echo "[Spatial_MAS] Running: python run_eval.py --models qwen llava gpt --split val $*"
python run_eval.py --models qwen llava gpt --split val "$@"

# 방금 생긴 results/ 폴더 찾기 (가장 최신)
LATEST_RUN=$(ls -td results/20* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
  echo "[Spatial_MAS] No results dir found. Skip analysis."
  exit 0
fi
echo "[Spatial_MAS] Latest run: $LATEST_RUN"

echo "[Spatial_MAS] Running analyze_failures.py ..."
python analyze_failures.py --run_dir "$LATEST_RUN"

echo "[Spatial_MAS] Running export_failed_samples.py ..."
python export_failed_samples.py --run_dir "$LATEST_RUN"

echo "[Spatial_MAS] Done. Results: $LATEST_RUN"
