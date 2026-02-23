#!/bin/bash
# 학습만 실행 (sample + train). 평가는 나중에 run_full_pipeline.sh 또는 03_evaluate.py로.
# Usage: bash scripts/sft_cvbench/run_train_all.sh [shots]
#   shots: 10 (default), 30, 100
set -e
cd "$(dirname "$0")/../.."
SHOTS=${1:-10}

echo "=============================================="
echo "SFT Training (CV-Bench) — shots=$SHOTS"
echo "=============================================="

# 1. Sample dataset
python scripts/sft_cvbench/01_sample_dataset.py

# 2. Train all models (spatialrgpt needs SPATIALRGPT_PATH)
for MODEL in qwen3_4b llava4d sa2va spatialreasoner; do
  echo ""
  echo "[Train] $MODEL..."
  python scripts/sft_cvbench/02_train.py --model $MODEL --shots $SHOTS || true
done

# spatialrgpt (optional)
if [ -n "$SPATIALRGPT_PATH" ] && [ -d "$SPATIALRGPT_PATH" ]; then
  echo ""
  echo "[Train] spatialrgpt..."
  python scripts/sft_cvbench/02_train.py --model spatialrgpt --shots $SHOTS || true
fi

echo ""
echo "=============================================="
echo "Done. Checkpoints: results/sft_cvbench/checkpoints/"
echo "=============================================="
