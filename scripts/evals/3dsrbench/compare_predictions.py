#!/usr/bin/env python3
"""
Compare predictions across 3DSRBench runs (Qwen3, Sa2VA, LLaVA4D).
Vérifie si les modèles produisent des prédictions identiques ou différentes.

Usage:
  python scripts/evals/3dsrbench/compare_predictions.py
  python scripts/evals/3dsrbench/compare_predictions.py --dir results/runs/3dsrbench
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_details(path: Path) -> dict:
    """Load details.jsonl, return {idx: pred}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["idx"]] = d.get("pred", "")
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare 3DSRBench predictions across models")
    parser.add_argument("--dir", default="results/runs/3dsrbench", help="Base dir (qwen3_4b/, sa2va/, llava4d/)")
    args = parser.parse_args()

    base = Path(args.dir)
    models = ["qwen3_4b", "sa2va", "llava4d"]
    runs = {}

    for m in models:
        subdirs = sorted((base / m).glob("*"), key=lambda p: p.name, reverse=True)
        if not subdirs:
            print(f"[!] No run found for {m}")
            continue
        latest = subdirs[0]
        details_path = latest / "details.jsonl"
        if not details_path.exists():
            print(f"[!] No details.jsonl in {latest}")
            continue
        runs[m] = load_details(details_path)
        print(f"[+] {m}: {latest.name} ({len(runs[m])} samples)")

    if len(runs) < 2:
        print("Need at least 2 model runs to compare.")
        return

    # Compare
    indices = sorted(set().union(*[set(r.keys()) for r in runs.values()]))
    n = len(indices)

    # Pairwise agreement
    print("\n--- Agreement (same pred) ---")
    for i, m1 in enumerate(models):
        if m1 not in runs:
            continue
        for m2 in models[i + 1 :]:
            if m2 not in runs:
                continue
            same = sum(1 for idx in indices if runs[m1].get(idx) == runs[m2].get(idx))
            pct = 100 * same / n if n else 0
            print(f"  {m1} vs {m2}: {same}/{n} identical ({pct:.1f}%)")

    # All three identical?
    if len(runs) == 3:
        all_same = sum(
            1 for idx in indices
            if runs["qwen3_4b"].get(idx) == runs["sa2va"].get(idx) == runs["llava4d"].get(idx)
        )
        print(f"\n  All 3 identical: {all_same}/{n} ({100*all_same/n:.1f}%)")

    # Sample differences
    diffs = []
    for idx in indices[:20]:  # first 20
        preds = {m: runs[m].get(idx, "?") for m in models if m in runs}
        if len(set(preds.values())) > 1:
            diffs.append((idx, preds))
    if diffs:
        print("\n--- Sample indices where models disagree ---")
        for idx, preds in diffs[:10]:
            print(f"  idx={idx}: {preds}")
    else:
        print("\n--- First 20 samples: all models agree ---")
        if indices:
            idx = indices[0]
            preds = {m: runs[m].get(idx, "?") for m in models if m in runs}
            print(f"  idx={idx}: {preds}")


if __name__ == "__main__":
    main()
