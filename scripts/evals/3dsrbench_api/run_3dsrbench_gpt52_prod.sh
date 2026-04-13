#!/usr/bin/env bash
# 3DSRBench (full HF test) + GPT-5.2 — production-style run
# - Loads .env if present (OPENAI_API_KEY, HF_TOKEN)
# - Retries: OPENAI_MAX_RETRIES (default 8) in runners.py
# - Checkpoint: --checkpoint_every (default 10)
#
# Usage:
#   cd /path/to/Spatial_MAS
#   bash scripts/evals/3dsrbench_api/run_3dsrbench_gpt52_prod.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or add to .env}"

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-8}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
EXTRA=(--prompt_variant with_prompt)
[[ "${PROMPT:-}" == "plain" ]] && EXTRA=(--without_prompt)
[[ "${BOTH_PROMPTS:-0}" == "1" ]] && EXTRA=()

echo "[prod] 3DSRBench full + gpt-5.2 | checkpoint_every=$CHECKPOINT_EVERY | max_tokens=$MAX_TOKENS"
python scripts/evals/3dsrbench_api/run_eval_api.py \
  --full_dataset \
  --model gpt_5_2 \
  --max_tokens "$MAX_TOKENS" \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  "${EXTRA[@]}"

echo "[prod] Done → results/runs/3dsrbench/api_models/full_dataset/"
