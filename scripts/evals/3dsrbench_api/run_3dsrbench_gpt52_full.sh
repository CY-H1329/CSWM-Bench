#!/usr/bin/bash
# 3DSRBench FULL (HuggingFace ccvl/3DSRBench test) — GPT-5.2 vision via OpenAI API
#
# Prérequis: OPENAI_API_KEY, dépôt Spatial_MAS, datasets HF en cache
#
# Usage:
#   cd /path/to/Spatial_MAS
#   bash scripts/evals/3dsrbench_api/run_3dsrbench_gpt52_full.sh
#
# Variantes:
#   PROMPT=plain bash ... # question seule (sans prompt spatial)
#   MAX_TOKENS=2048 bash ...

set -euo pipefail
cd "$(dirname "$0")/../../.."

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
EXTRA=(--prompt_variant with_prompt)
if [[ "${PROMPT:-}" == "plain" ]]; then
  EXTRA=(--without_prompt)
fi
# BOTH_PROMPTS=1 → with + without (2× coût API)
if [[ "${BOTH_PROMPTS:-0}" == "1" ]]; then
  EXTRA=()
fi

echo "[3DSRBench] Full dataset + GPT-5.2 (max_tokens=$MAX_TOKENS, extra=${EXTRA[*]:-both variants})"
python scripts/evals/3dsrbench_api/run_eval_api.py \
  --full_dataset \
  --model gpt_5_2 \
  --max_tokens "$MAX_TOKENS" \
  "${EXTRA[@]}"

echo "Done. See results/runs/3dsrbench/api_models/full_dataset/"
