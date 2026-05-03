#!/usr/bin/env bash
# Évaluation SpatiO (MAS v2) sur MindCube — prévu pour un serveur GPU type H100.
# Premier run Hugging Face : téléchargement ~600 Mo (data.zip MindCube).
#
# Usage :
#   cd /chemin/vers/Spatial_MAS
#   MAX_SAMPLES=50 bash scripts/evals/mindcube/run_mindcube_mas_v2_h100.sh
#
# Variables utiles :
#   CUDA_VISIBLE_DEVICES  (défaut : 0)
#   HF_HOME               cache Hugging Face
#   MINDCUBE_SPLIT        ex. train, test (voir carte du jeu HF)
#   OUTPUT_DIR            dossier résultats (défaut : results/mas_v2/mindcube/h100)
#   MAX_SAMPLES           nombre d’échantillons (défaut : 32)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MINDCUBE_SPLIT="${MINDCUBE_SPLIT:-train}"

MAX_SAMPLES="${MAX_SAMPLES:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mas_v2/mindcube/h100}"

python run_eval_mas_v2.py \
  --benchmark mindcube \
  --test_only \
  --max_samples "${MAX_SAMPLES}" \
  --device cuda \
  --use_local_reasoning \
  --low_memory \
  --output_dir "${OUTPUT_DIR}"

echo "Fichiers de temps : dans le sous-dossier horodaté, mas_timing.jsonl + mas_timing.log"
