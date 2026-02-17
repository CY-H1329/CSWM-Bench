#!/bin/bash
# Run Multi-Agent System evaluation on H100
# Usage:
#   bash scripts/run_h100_mas.sh                    # default: 3dsrbench, qwen_qwen_qwen
#   bash scripts/run_h100_mas.sh cvbench qwen llava qwen
#   bash scripts/run_h100_mas.sh 3dsrbench qwen qwen qwen --max_per_category 10

set -e
cd "$(dirname "$0")/.."

# Load .env if exists
[ -f .env ] && source .env

BENCHMARK="${1:-3dsrbench}"
HEAD="${2:-qwen3_4b}"
PERCEPTION="${3:-qwen3_4b}"
REASONING="${4:-qwen3_4b}"
EXTRA_ARGS=()
if [ $# -ge 4 ]; then
  shift 4
  EXTRA_ARGS=("$@")
fi

echo "[MAS] Benchmark=$BENCHMARK, Head=$HEAD, Perception=$PERCEPTION, Reasoning=$REASONING"
python run_eval_mas.py \
  --benchmark "$BENCHMARK" \
  --head "$HEAD" \
  --perception "$PERCEPTION" \
  --reasoning "$REASONING" \
  "${EXTRA_ARGS[@]}"
