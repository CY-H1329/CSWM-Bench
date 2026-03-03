#!/bin/bash
# CV-Bench 400 재샘플링: 카테고리별 정확히 100개 (Count, Relation, Depth, Distance)
# GitHub에 올리기 전 실행
#
# Usage:
#   bash scripts/resample_cvbench_400.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Re-sampling cvbench_400: 100 per category"
echo "  Count, Relation, Depth, Distance"
echo "=============================================="

python scripts/prepare_frozen_benchmarks.py --benchmarks cvbench_400

echo ""
echo "Done. Verify:"
echo "  cat data/frozen_benchmarks/cvbench_400/manifest.json"
echo ""
echo "GitHub에 올리기:"
echo "  git add data/frozen_benchmarks/cvbench_400/"
echo "  git commit -m 'Re-sample cvbench_400: 100 per category (Count, Relation, Depth, Distance)'"
echo "  git push"
