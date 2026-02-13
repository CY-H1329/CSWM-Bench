#!/usr/bin/env python3
"""
Télécharge tous les benchmarks une fois et les met en cache.
À exécuter sur H100 (ou local) une seule fois.
Les datasets HuggingFace sont mis en cache dans ~/.cache/huggingface/datasets/
et réutilisés automatiquement ensuite.

Usage:
  python scripts/setup_datasets.py
  python scripts/setup_datasets.py --benchmarks stvqa7k omni3d  # subset
"""
import argparse
import os
from pathlib import Path

# Force cache directory (optionnel, pour centraliser sur H100)
# HF_DATASETS_CACHE=/path/to/cache python scripts/setup_datasets.py
CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))


BENCHMARKS = {
    "stvqa7k": {
        "name": "OX-PIXL/STVQA-7K",
        "split": "val",
        "desc": "STVQA-7K (Spatial VQA, 692 val)",
    },
    "omni3d": {
        "name": "dmarsili/Omni3D-Bench",
        "split": "train",
        "desc": "OMNI3D-BENCH (501 samples)",
    },
    "cvbench": {
        "name": "nyu-visionx/CV-Bench",
        "split": "test",
        "desc": "CV-Bench (2.6k samples)",
    },
    "3dsrbench": {
        "name": "ccvl/3DSRBench",
        "split": "test",
        "subset": "benchmark",
        "desc": "3DSRBench (5.1k samples)",
    },
}


def download_benchmark(key: str, cfg: dict) -> bool:
    from datasets import load_dataset

    name = cfg["name"]
    split = cfg["split"]
    subset = cfg.get("subset")
    desc = cfg.get("desc", name)

    print(f"\n[{key}] {desc}")
    print(f"  Loading {name} (split={split})...")

    try:
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)
        print(f"  OK: {len(ds)} samples")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download all benchmark datasets")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Benchmarks to download",
    )
    parser.add_argument("--cache-dir", default=None, help="Override HF cache dir")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        print(f"Cache: {args.cache_dir}")
    else:
        print(f"Cache: {CACHE_DIR}")

    ok = 0
    for key in args.benchmarks:
        if key in BENCHMARKS:
            if download_benchmark(key, BENCHMARKS[key]):
                ok += 1

    print(f"\nDone: {ok}/{len(args.benchmarks)} benchmarks downloaded.")
    print("Les datasets sont en cache. Les runs suivants les réutiliseront automatiquement.")


if __name__ == "__main__":
    main()
