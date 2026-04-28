#!/usr/bin/env python3
"""
Analyze Stage1 vs Stage3 results and produce:
- results.json (merged)
- results_table.csv (per-task accuracies + gaps)
- failure_cases.jsonl (subset where stage1 correct but stage3 wrong, etc.)

This script expects:
  stage1_run_dir/results.json + details.jsonl
  stage3_run_dir/results.json + details.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", required=True)
    ap.add_argument("--stage3_run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    s1 = Path(args.stage1_run_dir)
    s3 = Path(args.stage3_run_dir)
    r1 = json.loads((s1 / "results.json").read_text(encoding="utf-8"))
    r3 = json.loads((s3 / "results.json").read_text(encoding="utf-8"))
    d1 = _read_jsonl(s1 / "details.jsonl")
    d3 = _read_jsonl(s3 / "details.jsonl")

    # Build per-task table
    tasks = sorted(set(list((r1.get("per_task") or {}).keys()) + list((r3.get("per_task") or {}).keys())))
    rows = []
    for t in tasks + ["total"]:
        if t == "total":
            a1 = float((r1.get("total") or {}).get("accuracy", 0.0))
            a3 = float((r3.get("total") or {}).get("accuracy", 0.0))
            n1 = int((r1.get("total") or {}).get("n", 0))
            n3 = int((r3.get("total") or {}).get("n", 0))
        else:
            a1 = float((r1.get("per_task") or {}).get(t, {}).get("accuracy", 0.0))
            a3 = float((r3.get("per_task") or {}).get(t, {}).get("accuracy", 0.0))
            n1 = int((r1.get("per_task") or {}).get(t, {}).get("n", 0))
            n3 = int((r3.get("per_task") or {}).get(t, {}).get("n", 0))
        rows.append({"task_type": t, "n_stage1": n1, "acc_stage1": a1, "n_stage3": n3, "acc_stage3": a3, "gap": a1 - a3})

    df = pd.DataFrame(rows)

    # Failure cases: stage1 correct but stage3 wrong, plus any stage3 wrong
    m1 = {x["sample_id"]: x for x in d1}
    m3 = {x["sample_id"]: x for x in d3}
    failures = []
    for sid, x3 in m3.items():
        x1 = m1.get(sid, {})
        gt = x3.get("gt", x1.get("gt", ""))
        p1 = x1.get("pred", "")
        p3 = x3.get("pred", "")
        s1_ok = (p1 == gt) if gt else False
        s3_ok = (p3 == gt) if gt else False
        if (s1_ok and not s3_ok) or (not s3_ok):
            failures.append(
                {
                    "sample_id": sid,
                    "task_type": x3.get("task_type", x1.get("task_type", "")),
                    "stage1_answer": p1,
                    "stage3_answer": p3,
                    "correct_answer": gt,
                    "wm_frames": x3.get("wm_frames", []),
                }
            )

    out_dir = Path(args.out_dir) if args.out_dir else s3.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "stage1_run_dir": s1.as_posix(),
        "stage3_run_dir": s3.as_posix(),
        "stage1_accuracy": {"total": float((r1.get("total") or {}).get("accuracy", 0.0)), **{k: float(v.get("accuracy", 0.0)) for k, v in (r1.get("per_task") or {}).items()}},
        "stage3_accuracy": {"total": float((r3.get("total") or {}).get("accuracy", 0.0)), **{k: float(v.get("accuracy", 0.0)) for k, v in (r3.get("per_task") or {}).items()}},
        "gap": {row["task_type"]: float(row["gap"]) for row in rows},
        "failure_cases_n": len(failures),
    }

    (out_dir / "results.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "results_table.csv", index=False)
    with open(out_dir / "failure_cases.jsonl", "w", encoding="utf-8") as f:
        for x in failures[:200]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()

