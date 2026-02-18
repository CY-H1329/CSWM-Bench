#!/bin/bash
# Spatial MAS Full Pipeline — H100 execution
# Head (GPT-5.2) → 3 Specialists (GPU + API) → Reasoning (DeepSeek-VL)
#
# Usage:
#   bash scripts/run_h100_mas_full.sh                    # cvbench, 100 samples
#   bash scripts/run_h100_mas_full.sh --test             # 5 samples
#   bash scripts/run_h100_mas_full.sh --max_samples 50  # 50 samples
#   bash scripts/run_h100_mas_full.sh --full_dataset    # full dataset
#   bash scripts/run_h100_mas_full.sh 3dsrbench --test  # 3DSRBench, test
#
# Env (required): OPENAI_API_KEY, DEEPSEEK_API_KEY
# Env (for API specialists): ANTHROPIC_API_KEY, GEMINI_API_KEY
# GPU: qwen3_4b, sa2va, llava4d run on H100
set -e
cd "$(dirname "$0")/.."

[ -f .env ] && source .env

BENCHMARK="cvbench"
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == "cvbench" || "$arg" == "3dsrbench" ]]; then
    BENCHMARK="$arg"
  else
    EXTRA+=("$arg")
  fi
done

# TF32, Flash Attention for H100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[MAS Full] Benchmark=$BENCHMARK"
echo "  Head: GPT-5.2 (API)"
echo "  Specialists: qwen3_4b, sa2va, llava4d (GPU) + claude, gpt4o, gemini (API)"
echo "  Reasoning: DeepSeek-VL (GPU, open-source)"
echo ""

python scripts/evals/mas_pipeline/run_eval_mas.py \
  --benchmark "$BENCHMARK" \
  "${EXTRA[@]}"
