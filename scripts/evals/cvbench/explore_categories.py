#!/usr/bin/env python3
"""
Explore CV-Bench task types and type (2D/3D) distribution.

Usage:
  python scripts/evals/cvbench/explore_categories.py
  python scripts/evals/cvbench/explore_categories.py --max_samples 500
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarks import load_benchmark


def main():
    parser = argparse.ArgumentParser(description="Explore CV-Bench categories")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading CV-Bench...")
    ds = load_benchmark("cvbench", max_samples=args.max_samples, seed=args.seed)
    print(f"  {len(ds)} samples\n")

    tasks = []
    types = []
    sources = []
    for i in range(len(ds)):
        ex = ds[i]
        tasks.append(ex.get("task") or "unknown")
        types.append(ex.get("type") or "unknown")
        sources.append(ex.get("source") or "unknown")

    print("=== Task distribution ===")
    for task, count in sorted(Counter(tasks).items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")

    print("\n=== Type (2D/3D) distribution ===")
    for t, count in sorted(Counter(types).items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    print("\n=== Source distribution ===")
    for s, count in sorted(Counter(sources).items(), key=lambda x: -x[1]):
        print(f"  {s}: {count}")

    print("\n=== All unique tasks ===")
    print(sorted(set(tasks)))


if __name__ == "__main__":
    main()
