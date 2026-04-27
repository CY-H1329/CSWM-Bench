#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from predictors import OraclePredictor, RandomPredictor, LearnedMLPPredictor


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _score(ex: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    task = ex.get("task")
    gt = ex.get("gt", {}) or {}
    s = {"id": ex.get("id"), "task": task}

    if task == "A":
        s["divergence_correct"] = str(pred.get("divergence", "")).lower().strip() == str(gt.get("divergence", "")).lower().strip()
        s["outcomes_correct"] = (
            str(pred.get("case1_outcome", "")).lower().strip() == str(gt.get("case1_outcome", "")).lower().strip()
            and str(pred.get("case2_outcome", "")).lower().strip() == str(gt.get("case2_outcome", "")).lower().strip()
        )
        s["reason_correct"] = str(pred.get("reason_label", "")).strip() == str(gt.get("reason_label", "")).strip()
        s["all_correct"] = s["divergence_correct"] and s["outcomes_correct"] and s["reason_correct"]
        return s

    if task == "B":
        s["divergence_correct"] = str(pred.get("divergence_label", "")).strip() == str(gt.get("divergence_label", "")).strip()
        s["reason_correct"] = str(pred.get("reason_label", "")).strip() == str(gt.get("reason_label", "")).strip()
        s["all_correct"] = s["divergence_correct"] and s["reason_correct"]
        return s

    s["error"] = "unknown_task"
    return s


def _mean_bool(xs: List[bool]) -> float:
    return (sum(1 for x in xs if x) / len(xs)) if xs else 0.0


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task = defaultdict(list)
    for s in scores:
        by_task[s.get("task", "unknown")].append(s)
    overall = {
        "n": len(scores),
        "divergence_accuracy": _mean_bool([s.get("divergence_correct", False) for s in scores]),
        "reason_accuracy": _mean_bool([s.get("reason_correct", False) for s in scores]),
        "all_correct": _mean_bool([s.get("all_correct", False) for s in scores]),
    }
    per_task = {}
    for t, rows in sorted(by_task.items()):
        per_task[t] = {
            "n": len(rows),
            "divergence_accuracy": _mean_bool([r.get("divergence_correct", False) for r in rows]),
            "reason_accuracy": _mean_bool([r.get("reason_correct", False) for r in rows]),
            "all_correct": _mean_bool([r.get("all_correct", False) for r in rows]),
        }
    return {"overall": overall, "per_task": per_task}


def main() -> None:
    parser = argparse.ArgumentParser(description="CSWM-Bench WM track eval")
    parser.add_argument("--data", default=str(ROOT / "data" / "cswmbench_wm" / "cswmbench_wm.jsonl"))
    parser.add_argument("--predictor", choices=["oracle", "random", "mlp"], default="oracle")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default=str(ROOT / "results" / "runs" / "cswmbench_wm_models" / "mlp_multihead" / "ckpt.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = _read_jsonl(Path(args.data))
    if args.max_samples:
        data = data[: min(args.max_samples, len(data))]

    if args.predictor == "oracle":
        pred = OraclePredictor()
    elif args.predictor == "random":
        pred = RandomPredictor(seed=args.seed)
    else:
        pred = LearnedMLPPredictor(checkpoint_path=args.ckpt, device=args.device)

    details = []
    scores = []
    for ex in data:
        p = pred.predict(ex)
        sc = _score(ex, p)
        scores.append(sc)
        details.append({"id": ex.get("id"), "task": ex.get("task"), "gt": ex.get("gt"), "pred": p, "score": sc})

    metrics = _aggregate(scores)
    out_dir = ROOT / "results" / "runs" / "cswmbench_wm" / datetime.now().strftime("%Y%m%d_%H%M%S") / args.predictor
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Résultats: {out_dir}")


if __name__ == "__main__":
    main()

