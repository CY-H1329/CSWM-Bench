#!/bin/bash
# STVQA-7K Evaluation — H100
# Dataset: hunarbatra/STVQA-7K val (692 samples)
# Models: Qwen-3.0-VL 4B, LLaVA-4D, SpatialReasoner, SpatialRGPT, Sa2VA
#
# Usage on H100:
#   git pull
#   cd Spatial_MAS
#   export SPATIALRGPT_PATH=/path/to/SpatialRGPT   # required for spatialrgpt
#   bash experiments/stvqa7k/run_h100.sh
#
# Optional: --max_samples 10 for quick test
set -e

export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::UserWarning"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-results/stvqa7k}"
MAX_SAMPLES="${MAX_SAMPLES:-692}"

echo "=============================================="
echo "STVQA-7K Evaluation — H100"
echo "Project: $PROJECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Patch SpatialRGPT for Python 3.9 (match statement requires 3.10+)
python scripts/stvqa7k/patch_spatialrgpt_py39.py || true

for MODEL in qwen3_4b llava4d sa2va spatialreasoner spatialrgpt; do
  echo ""
  echo ">>> $MODEL"
  echo "----------------------------------------------"
  if [[ -n "$MAX_SAMPLES" ]]; then
    python scripts/stvqa7k/eval_stvqa7k.py --model "$MODEL" --output_dir "$OUTPUT_DIR" --max_samples "$MAX_SAMPLES"
  else
    python scripts/stvqa7k/eval_stvqa7k.py --model "$MODEL" --output_dir "$OUTPUT_DIR"
  fi
done

echo ""
echo "=============================================="
echo "Done. Results in $OUTPUT_DIR/"
echo "  all_results.json — summary"
echo "  <model>/results.json — per-model"
echo "=============================================="
