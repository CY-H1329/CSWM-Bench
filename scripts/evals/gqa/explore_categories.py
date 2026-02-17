#!/usr/bin/env python3
"""
Explore GQA dataset categories.
Run to find unique semantic/structural categories before full evaluation.

Usage:
  python scripts/evals/gqa/explore_categories.py
  python scripts/evals/gqa/explore_categories.py --max_samples 5000
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarks import load_benchmark, get_benchmark_category


def main():
    parser = argparse.ArgumentParser(description="Explore GQA categories")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading GQA (val_balanced)...")
    ds = load_benchmark("gqa", max_samples=args.max_samples, seed=args.seed)
    print(f"  {len(ds)} samples")

    semantic = Counter()
    structural = Counter()
    for i in range(len(ds)):
        ex = ds[i]
        sem = ex.get("semantic") or ex.get("category") or "unknown"
        struct = ex.get("structural") or "unknown"
        semantic[sem] += 1
        structural[struct] += 1

    print("\n=== Semantic categories ===")
    for cat, n in semantic.most_common():
        print(f"  {cat}: {n}")

    print("\n=== Structural categories ===")
    for cat, n in structural.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
