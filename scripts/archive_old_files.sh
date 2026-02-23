#!/bin/bash
# Archive old files (before baseline + MAS) on H100 server.
# Keeps: baseline (single model eval), MAS pipeline, 3dsrbench_api (runners for MAS)
# Moves to _archive/: old eval scripts, head_agent experiments, obsolete run scripts
#
# Usage:
#   bash scripts/archive_old_files.sh          # run
#   bash scripts/archive_old_files.sh --dry-run  # preview only

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
ARCHIVE="$ROOT/_archive"
DRY_RUN=false
[ "$1" = "--dry-run" ] && DRY_RUN=true

maybe_mv() {
  if [ -e "$1" ]; then
    if $DRY_RUN; then
      echo "  [dry-run] would move: $1 -> $2"
    else
      mkdir -p "$(dirname "$2")"
      mv "$1" "$2"
      echo "  archived: $1"
    fi
  fi
}

echo "=== Archiving old files to _archive/ ==="
$DRY_RUN && echo "(dry-run: no files moved)" && echo ""

# 1. Old root-level eval scripts (replaced by scripts/evals/...)
for f in run_eval.py run_eval_collab.py run_eval_unified.py run_eval_multiagent.py run_eval_mas.py run_eval_mas_full.py; do
  if [ -f "$ROOT/$f" ]; then
    $DRY_RUN || mv "$ROOT/$f" "$ARCHIVE/$f"
    echo "  archived: $f"
  fi
done

# 2. Old run scripts (keep run_h100_mas_full.sh, run_h100_setup_datasets.sh)
mkdir -p "$ARCHIVE/scripts"
for f in run_h100.sh run_h100_collab.sh run_h100_multiagent.sh run_h100_mas.sh; do
  if [ -f "$ROOT/scripts/$f" ]; then
    $DRY_RUN || mv "$ROOT/scripts/$f" "$ARCHIVE/scripts/$f"
    echo "  archived: scripts/$f"
  fi
done

# 3. Head-Agent experiments (exploratory, before current MAS)
if [ -d "$ROOT/scripts/evals/head_agent_cvbench" ]; then
  mkdir -p "$ARCHIVE/scripts/evals"
  $DRY_RUN || mv "$ROOT/scripts/evals/head_agent_cvbench" "$ARCHIVE/scripts/evals/"
  echo "  archived: scripts/evals/head_agent_cvbench/"
fi

# 4. Head-Agent results + summary
mkdir -p "$ARCHIVE/results_summary"
if [ -d "$ROOT/results_summary/head_agent" ]; then
  $DRY_RUN || mv "$ROOT/results_summary/head_agent" "$ARCHIVE/results_summary/"
  echo "  archived: results_summary/head_agent/"
fi
if [ -f "$ROOT/results_summary/HEAD_AGENT_SUMMARY.md" ]; then
  $DRY_RUN || mv "$ROOT/results_summary/HEAD_AGENT_SUMMARY.md" "$ARCHIVE/results_summary/"
  echo "  archived: results_summary/HEAD_AGENT_SUMMARY.md"
fi

# 5. Utility scripts (analyze_failures, export_failed_samples)
for f in analyze_failures.py export_failed_samples.py; do
  if [ -f "$ROOT/$f" ]; then
    $DRY_RUN || mv "$ROOT/$f" "$ARCHIVE/$f"
    echo "  archived: $f"
  fi
done

# 6. Head-Agent summarize script
if [ -f "$ROOT/scripts/summarize_head_agent_results.py" ]; then
  $DRY_RUN || mv "$ROOT/scripts/summarize_head_agent_results.py" "$ARCHIVE/scripts/"
  echo "  archived: scripts/summarize_head_agent_results.py"
fi

# 7. Old docs (head_agent, push_pull variants)
mkdir -p "$ARCHIVE/docs"
for f in HEAD_AGENT_SELECTION_PROTOCOL.md HEAD_AGENT_PUSH_PULL_WORKFLOW.md PULL_PUSH_간단정리.md PUSH_PULL_WORKFLOW.md; do
  if [ -f "$ROOT/docs/$f" ]; then
    $DRY_RUN || mv "$ROOT/docs/$f" "$ARCHIVE/docs/"
    echo "  archived: docs/$f"
  fi
done

# 8. docs/experiments (baseline_experiments planning docs)
if [ -d "$ROOT/docs/experiments" ]; then
  $DRY_RUN || mv "$ROOT/docs/experiments" "$ARCHIVE/docs/"
  echo "  archived: docs/experiments/"
fi

# 9. gather_results_summary references head_agent - keep but it may need adjustment
# (leave gather_results_summary.py - it's useful for MAS too)

echo ""
echo "=== Done. Archived to $ARCHIVE ==="
echo ""
echo "KEPT (baseline + MAS):"
echo "  - run_eval_single_3dsrbench.py"
echo "  - scripts/evals/3dsrbench/ (GPU single model)"
echo "  - scripts/evals/3dsrbench_api/ (API + runners for MAS)"
echo "  - scripts/evals/mas_pipeline/"
echo "  - scripts/evals/compare_qwen3_mas.py"
echo "  - scripts/evals/export_mas_scores_to_excel.py"
echo "  - scripts/run_h100_mas_full.sh"
echo "  - src/agents/mas/"
echo "  - configs/mas/"
