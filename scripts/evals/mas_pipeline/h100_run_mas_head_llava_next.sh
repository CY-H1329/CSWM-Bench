#!/usr/bin/env bash
# =============================================================================
# H100 — Spatial MAS, Head LLaVA-NeXT-7B, CV-Bench FULL (split HuggingFace test)
#
# Version : mas_pipeline_version head-llava-next-7b-cvbench-full-hf-2026.04.03
# Sortie   : results/runs/mas_pipeline/full_cvbench_hf/
#
#   cd ~/CY/Spatial_MAS/Spatial_MAS/Spatial_MAS
#   git pull origin main
#   conda activate spatial_reasoning
#   bash scripts/evals/mas_pipeline/h100_run_mas_head_llava_next.sh
#
# Test rapide (5 échantillons, pas le full) :
#   python scripts/evals/mas_pipeline/run_eval_mas.py \
#     --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml --test
#
# Sous-échantillon (sans full HF), ex. 100 :
#   python scripts/evals/mas_pipeline/run_eval_mas.py \
#     --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml \
#     --benchmark cvbench --max_samples 100
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[H100] repo: $REPO_ROOT"
echo "[H100] CV-Bench FULL (HF test split) | Head = LLaVA-NeXT-7B | out: mas_pipeline/full_cvbench_hf/"

exec python3 scripts/evals/mas_pipeline/run_eval_mas.py \
  --config scripts/evals/mas_pipeline/config_mas_head_llava_next.yaml \
  --benchmark cvbench \
  --full_dataset
