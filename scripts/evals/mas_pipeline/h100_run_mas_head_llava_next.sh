#!/usr/bin/env bash
# =============================================================================
# H100 — Spatial MAS avec Head-Agent LLaVA-NeXT-7B (GPU)
#
# Version config : mas_pipeline_version head-llava-next-7b-2026.04.02
# (voir config_mas_head_llava_next.yaml)
#
# Prérequis : repo à jour, CUDA, transformers, torch, clé HF si gated (souvent non)
#
#   cd ~/CY/Spatial_MAS/Spatial_MAS/Spatial_MAS   # racine avec src/ et scripts/
#   git pull origin main
#   conda activate spatial_reasoning   # ou votre env
#   bash scripts/evals/mas_pipeline/h100_run_mas_head_llava_next.sh
#
# Test rapide (5 samples) :
#   python scripts/evals/mas_pipeline/run_eval_mas.py --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml --test
#
# Full CV-Bench (attention VRAM / temps) :
#   python scripts/evals/mas_pipeline/run_eval_mas.py --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml --full_dataset
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[H100] repo: $REPO_ROOT"
echo "[H100] config: config_mas_head_llava_next.yaml (Head = LLaVA-NeXT-7B Mistral HF)"

exec python3 scripts/evals/mas_pipeline/run_eval_mas.py \
  --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml \
  --max_samples "${MAX_SAMPLES:-50}"
