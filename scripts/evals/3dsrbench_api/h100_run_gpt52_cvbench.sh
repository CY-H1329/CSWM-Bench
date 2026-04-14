#!/usr/bin/env bash
# =============================================================================
# H100 / Linux — CV-Bench (full HF test) + GPT-5.2
#
#   bash scripts/evals/3dsrbench_api/h100_run_gpt52_cvbench.sh
#
# Optional: CHECKPOINT_EVERY, MAX_TOKENS, HF_TOKEN — same as 3DSRBench script.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or add to .env}"

if ! python3 -c "import datasets, openai" 2>/dev/null; then
  pip install -q datasets huggingface_hub openai pyyaml tqdm pillow requests
fi

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-8}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

echo "[H100] CV-Bench full + gpt_5_2 | checkpoint_every=$CHECKPOINT_EVERY"

exec python3 scripts/evals/3dsrbench_api/run_eval_api.py \
  --benchmark cvbench \
  --full_dataset \
  --model gpt_5_2 \
  --prompt_variant with_prompt \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --max_tokens "$MAX_TOKENS"
