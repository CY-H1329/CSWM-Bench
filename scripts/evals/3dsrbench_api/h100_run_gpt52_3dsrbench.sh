#!/usr/bin/env bash
# =============================================================================
# H100 / Linux server — 3DSRBench (full HF test) + GPT-5.2 (OpenAI API)
#
# 1) Put repo on server (pick one):
#    git clone https://github.com/CY-H1329/Spatial_MAS.git && cd Spatial_MAS
#    git pull origin main
#
# 2) Run inside tmux (survives laptop disconnect):
#    tmux new -s gpt52_3dsr
#    export OPENAI_API_KEY="sk-..."
#    bash scripts/evals/3dsrbench_api/h100_run_gpt52_3dsrbench.sh
#    Detach: Ctrl+B then D  |  Reattach: tmux attach -t gpt52_3dsr
#
# Optional env: HF_TOKEN, OPENAI_MAX_RETRIES, CHECKPOINT_EVERY, MAX_TOKENS
# Or put OPENAI_API_KEY in .env at repo root.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[H100] repo: $REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or add to .env}"

if ! python3 -c "import datasets, openai" 2>/dev/null; then
  echo "[H100] pip install (minimal)..."
  pip install -q datasets huggingface_hub openai pyyaml tqdm pillow requests
fi

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-8}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

echo "[H100] 3DSRBench full + gpt_5_2 | checkpoint_every=$CHECKPOINT_EVERY | max_tokens=$MAX_TOKENS"

exec python3 scripts/evals/3dsrbench_api/run_eval_api.py \
  --full_dataset \
  --model gpt_5_2 \
  --prompt_variant with_prompt \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --max_tokens "$MAX_TOKENS"
