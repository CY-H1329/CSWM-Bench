#!/usr/bin/env python3
"""
Prepare frozen benchmark datasets for paper experiments.

Creates stratified samples (even across task categories) and saves to data/frozen_benchmarks/.
These datasets MUST NOT be modified - they are used for all paper experiments.

Benchmarks:
  - CV-Bench: 400 samples (100 per category: Count, Relation, Depth, Distance)
  - 3DSRBench: 500 samples (~42 per category across 12 categories)
  - STVQA: full val split (692 samples, no sampling)

Usage:
  python scripts/prepare_frozen_benchmarks.py
  python scripts/prepare_frozen_benchmarks.py --output-dir data/frozen_benchmarks
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


FROZEN_CONFIG = {
    "cvbench_400": {
        "source": "nyu-visionx/CV-Bench",
        "split": "test",
        "subset": None,
        "category_key": "task",
        "n_total": 400,
        "n_per_category": 100,  # 4 categories: Count, Relation, Depth, Distance
        "categories": ["Count", "Relation", "Depth", "Distance"],
    },
    "3dsrbench_500": {
        "source": "ccvl/3DSRBench",
        "split": "test",
        "subset": "benchmark",
        "category_key": "category",
        "n_total": 500,
        "n_per_category": None,  # Stratified: 500/12 ≈ 41-42 per category
        "categories": [
            "height_higher", "location_above", "location_closer_to_camera",
            "location_next_to", "multi_object_closer_to", "multi_object_facing",
            "multi_object_parallel", "multi_object_same_direction",
            "multi_object_viewpoint_towards_object", "orientation_in_front_of",
            "orientation_on_the_left", "orientation_viewpoint",
        ],
    },
    "stvqa_full": {
        "source": "hunarbatra/STVQA-7K",
        "split": "val",
        "subset": None,
        "category_key": "category",
        "n_total": None,  # Full eval - no sampling
        "n_per_category": None,
        "categories": None,
    },
}

SEED = 42


def stratified_sample_indices(ds, category_key: str, n_total: int, seed: int):
    """Sample n_total indices with equal count per category (stratified)."""
    rng = random.Random(seed)
    by_cat = {}
    for i in range(len(ds)):
        c = ds[i].get(category_key) or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    n_cats = len(by_cat)
    base_per_cat = n_total // n_cats
    remainder = n_total % n_cats

    indices = []
    for c in sorted(by_cat.keys()):
        idx_list = by_cat[c]
        # First 'remainder' categories get base+1, rest get base
        k = base_per_cat + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        k = min(k, len(idx_list))
        indices.extend(rng.sample(idx_list, k))

    indices.sort()
    return indices


def prepare_cvbench_400(cfg: dict, output_dir: Path, seed: int) -> dict:
    """CV-Bench: 100 per category (4 categories) = 400 total."""
    print("\n[CV-Bench 400] Loading...")
    ds = load_dataset(cfg["source"], split=cfg["split"])
    cat_key = cfg["category_key"]

    by_cat = {}
    for i in range(len(ds)):
        c = ds[i].get(cat_key) or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    rng = random.Random(seed)
    indices = []
    for c in cfg["categories"]:
        if c not in by_cat:
            print(f"  WARNING: category '{c}' not found, skipping")
            continue
        idx_list = by_cat[c]
        k = min(cfg["n_per_category"], len(idx_list))
        indices.extend(rng.sample(idx_list, k))
        print(f"  {c}: {k}/{len(idx_list)}")

    indices.sort()
    sampled = ds.select(indices)
    out_path = output_dir / "cvbench_400"
    out_path.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(str(out_path))

    manifest = {
        "benchmark": "cvbench",
        "n_samples": len(sampled),
        "n_per_category": cfg["n_per_category"],
        "categories": cfg["categories"],
        "seed": seed,
        "source": cfg["source"],
        "split": cfg["split"],
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(sampled)} samples to {out_path}")
    return manifest


def prepare_3dsrbench_500(cfg: dict, output_dir: Path, seed: int) -> dict:
    """3DSRBench: 500 samples stratified across 12 categories."""
    print("\n[3DSRBench 500] Loading...")
    ds = load_dataset(cfg["source"], cfg["subset"], split=cfg["split"])
    cat_key = cfg["category_key"]

    indices = stratified_sample_indices(ds, cat_key, cfg["n_total"], seed)

    # Report per-category counts
    by_cat = {}
    for i in indices:
        c = ds[i].get(cat_key) or "unknown"
        by_cat[c] = by_cat.get(c, 0) + 1
    for c in sorted(by_cat.keys()):
        print(f"  {c}: {by_cat[c]}")

    sampled = ds.select(indices)
    out_path = output_dir / "3dsrbench_500"
    out_path.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(str(out_path))

    manifest = {
        "benchmark": "3dsrbench",
        "n_samples": len(sampled),
        "categories": list(by_cat.keys()),
        "seed": seed,
        "source": cfg["source"],
        "subset": cfg["subset"],
        "split": cfg["split"],
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(sampled)} samples to {out_path}")
    return manifest


def prepare_stvqa_full(cfg: dict, output_dir: Path, seed: int) -> dict:
    """STVQA: full val split (692 samples), no sampling."""
    print("\n[STVQA full] Loading...")
    ds = load_dataset(cfg["source"], split=cfg["split"])

    out_path = output_dir / "stvqa_full"
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    manifest = {
        "benchmark": "stvqa",
        "n_samples": len(ds),
        "source": cfg["source"],
        "split": cfg["split"],
        "sampling": "none (full eval)",
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(ds)} samples to {out_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen benchmark datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "frozen_benchmarks",
        help="Output directory for frozen benchmarks",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["cvbench_400", "3dsrbench_500", "stvqa_full"],
        choices=["cvbench_400", "3dsrbench_500", "stvqa_full"],
        help="Which benchmarks to prepare",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    readme = {
        "frozen_benchmarks": "DO NOT MODIFY - Used for all paper experiments",
        "seed": args.seed,
        "paths": {
            "cvbench": "cvbench_400",
            "3dsrbench": "3dsrbench_500",
            "stvqa": "stvqa_full",
        },
    }
    (args.output_dir / "README.json").write_text(json.dumps(readme, indent=2))

    for key in args.benchmarks:
        cfg = FROZEN_CONFIG[key]
        if key == "cvbench_400":
            prepare_cvbench_400(cfg, args.output_dir, args.seed)
        elif key == "3dsrbench_500":
            prepare_3dsrbench_500(cfg, args.output_dir, args.seed)
        elif key == "stvqa_full":
            prepare_stvqa_full(cfg, args.output_dir, args.seed)

    print("\nDone. Frozen benchmarks ready at", args.output_dir)


if __name__ == "__main__":
    main()
