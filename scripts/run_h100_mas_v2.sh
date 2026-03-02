#!/bin/bash
# MAS v2 — H100 서버 실행 스크립트
#
# SpatialRGPT, Sa2VA는 구버전 env 필요 → H100에서 충돌 가능.
# 해결: --specialist_whitelist 로 3-agent 모드 사용 (qwen3_4b, llava4d, spatial_reasoner)
#
# Usage:
#   # 3-agent (권장, H100 호환)
#   bash scripts/run_h100_mas_v2.sh 3agent
#
#   # 5-agent (Sa2VA, SpatialRGPT 포함 — env 호환 시)
#   bash scripts/run_h100_mas_v2.sh 5agent
#
#   # Test only, 10 samples
#   bash scripts/run_h100_mas_v2.sh 3agent --test_only --max_samples 10
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-3agent}"
shift || true

# 3-agent: Sa2VA, SpatialRGPT 제외 (H100 호환)
WHITELIST_3="qwen3_4b,llava4d,spatial_reasoner"

# 5-agent: 전체 (Sa2VA/SpatialRGPT env 호환 필요)
WHITELIST_5=""

echo "=============================================="
echo "MAS v2 — H100"
echo "Mode: $MODE"
echo "=============================================="

if [[ "$MODE" == "3agent" ]]; then
  EXTRA_ARGS=("--specialist_whitelist" "$WHITELIST_3")
  echo "Specialists: qwen3_4b, llava4d, spatial_reasoner (Sa2VA/SpatialRGPT 제외)"
elif [[ "$MODE" == "5agent" ]]; then
  EXTRA_ARGS=()
  echo "Specialists: qwen3_4b, sa2va, llava4d, spatial_rgpt, spatial_reasoner"
  echo "  → Sa2VA/SpatialRGPT env 호환 필요. 실패 시 3agent 사용."
else
  echo "Usage: $0 {3agent|5agent} [--test_only --max_samples N] [--benchmark cvbench|3dsrbench] ..."
  exit 1
fi

# SpatialRGPT patch (5agent 시)
if [[ "$MODE" == "5agent" && -n "${SPATIALRGPT_PATH:-}" ]]; then
  python scripts/stvqa7k/patch_spatialrgpt_py39.py 2>/dev/null || true
fi

python run_eval_mas_v2.py \
  --benchmark cvbench \
  --use_vlm_reasoning \
  --specialist_offload_after_use \
  "${EXTRA_ARGS[@]}" \
  "$@"
