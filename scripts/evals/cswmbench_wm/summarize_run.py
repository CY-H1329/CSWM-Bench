#!/usr/bin/env python3
"""
Summarize a CSWM WM-track run into slide-ready Markdown.

Input: results/runs/cswmbench_wm/<timestamp>/<predictor>/results.json (+ optional details.jsonl)
Output: writes summary.md next to results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., results/runs/cswmbench_wm/20260427_074841/random")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing: {results_path}")

    metrics: Dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    overall = metrics.get("overall", {})
    per_task = metrics.get("per_task", {})

    md = []
    md.append(f"## CSWM WM-track — Résultats (`{run_dir.as_posix()}`)\n")
    md.append("### Summary\n")
    md.append(f"- **N**: {overall.get('n')}\n")
    md.append(f"- **Divergence Accuracy**: {overall.get('divergence_accuracy'):.4f}\n")
    md.append(f"- **Reason Accuracy**: {overall.get('reason_accuracy'):.4f}\n")
    md.append(f"- **All-correct**: {overall.get('all_correct'):.4f}\n")

    md.append("\n### Per task\n")
    for t in sorted(per_task.keys()):
        row = per_task[t]
        md.append(f"- **Task {t}** (n={row.get('n')}): div={row.get('divergence_accuracy'):.4f}, reason={row.get('reason_accuracy'):.4f}, all={row.get('all_correct'):.4f}\n")

    out_path = run_dir / "summary.md"
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

