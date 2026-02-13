#!/bin/bash
# Télécharge tous les benchmarks une fois. À exécuter sur H100 une seule fois.
# Les datasets sont mis en cache (~/.cache/huggingface/datasets) et réutilisés ensuite.

set -e
cd "$(dirname "$0")/.."

echo "=== Setup: téléchargement des 4 benchmarks ==="
python scripts/setup_datasets.py

echo ""
echo "Done. Les datasets sont en cache."
echo "Pour les runs suivants, pas besoin de re-télécharger."
