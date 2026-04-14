#!/usr/bin/env bash
# =============================================================================
# H100 / Linux — MMSI-Bench (RunsenXu/MMSI-Bench, 1000 samples, multi-image) + GPT-5.2
#
#   bash scripts/evals/3dsrbench_api/h100_run_gpt52_mmsibench.sh
#
# Requires: OPENAI_API_KEY, datasets, multi-image support in runners (Claude/GPT/Gemini).
# DeepSeek native vision stitches frames into one image when >1 frame.
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

MAX_TOKENS="${MAX_TOKENS:-2048}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

echo "[H100] MMSI-Bench full + gpt_5_2 | checkpoint_every=$CHECKPOINT_EVERY | max_tokens=$MAX_TOKENS"

exec python3 scripts/evals/3dsrbench_api/run_eval_api.py \
  --benchmark mmsibench \
  --full_dataset \
  --model gpt_5_2 \
  --prompt_variant with_prompt \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --max_tokens "$MAX_TOKENS"
