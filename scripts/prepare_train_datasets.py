#!/usr/bin/env python3
"""
Create train datasets from samples NOT used in frozen benchmarks.

- cvbench_train_300: 300 samples from CV-Bench test (excluding frozen 400)
- 3dsrbench_train_300: 300 samples from 3DSRBench (excluding frozen 500)
- stvqa_train_300: 300 samples from STVQA-7K train split

Uses same seed (42) as frozen to ensure disjoint sets.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FROZEN_DIR = ROOT / "data" / "frozen_benchmarks"


def get_frozen_cvbench_indices(ds, seed: int) -> set:
    """Replicate prepare_frozen_benchmarks CV-Bench sampling to get frozen indices."""
    cat_key = "task"
    categories = ["Count", "Relation", "Depth", "Distance"]
    n_per_category = 100

    by_cat = {}
    for i in range(len(ds)):
        c = ds[i].get(cat_key) or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    rng = random.Random(seed)
    indices = []
    for c in categories:
        if c not in by_cat:
            continue
        idx_list = by_cat[c]
        k = min(n_per_category, len(idx_list))
        indices.extend(rng.sample(idx_list, k))
    return set(indices)


def get_frozen_3dsrbench_indices(ds, seed: int) -> set:
    """Replicate stratified_sample_indices to get frozen 3DSRBench indices."""
    cat_key = "category"
    n_total = 500

    by_cat = {}
    for i in range(len(ds)):
        c = ds[i].get(cat_key) or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    n_cats = len(by_cat)
    base_per_cat = n_total // n_cats
    remainder = n_total % n_cats

    rng = random.Random(seed)
    indices = []
    for c in sorted(by_cat.keys()):
        idx_list = by_cat[c]
        k = base_per_cat + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        k = min(k, len(idx_list))
        indices.extend(rng.sample(idx_list, k))
    return set(indices)


def stratified_sample_from_pool(by_cat: dict, n_total: int, rng: random.Random) -> list:
    """Sample n_total from by_cat (category -> list of indices), stratified."""
    n_cats = len(by_cat)
    if n_cats == 0:
        return []
    base_per_cat = n_total // n_cats
    remainder = n_total % n_cats
    indices = []
    for c in sorted(by_cat.keys()):
        idx_list = by_cat[c]
        k = base_per_cat + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        k = min(k, len(idx_list))
        indices.extend(rng.sample(idx_list, k))
    indices.sort()
    return indices


def prepare_cvbench_train(output_dir: Path, seed: int, n_samples: int = 300) -> dict:
    """CV-Bench: n_samples stratified (excluding frozen 400 when possible)."""
    print(f"\n[CV-Bench train {n_samples}] Loading...")
    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    frozen_indices = get_frozen_cvbench_indices(ds, seed)
    remaining = [i for i in range(len(ds)) if i not in frozen_indices]
    print(f"  Full: {len(ds)}, Frozen: {len(frozen_indices)}, Remaining: {len(remaining)}")

    by_cat = {}
    pool = remaining if len(remaining) >= n_samples else list(range(len(ds)))
    if len(remaining) < n_samples:
        print(f"  [Warn] Remaining {len(remaining)} < {n_samples}, sampling from full (may overlap frozen)")
    for i in pool:
        c = ds[i].get("task") or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    rng = random.Random(seed + 1)
    indices = stratified_sample_from_pool(by_cat, n_samples, rng)
    sampled = ds.select(indices)

    out_path = output_dir / f"cvbench_train_{n_samples}"
    out_path.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(str(out_path))

    by_cat_out = {}
    for i in indices:
        c = ds[i].get("task") or "unknown"
        by_cat_out[c] = by_cat_out.get(c, 0) + 1
    manifest = {
        "benchmark": "cvbench",
        "split": "train",
        "n_samples": len(sampled),
        "excludes": "cvbench_400 (frozen)" if pool == remaining else "overlap (insufficient remaining)",
        "seed": seed,
        "source": "nyu-visionx/CV-Bench",
        "per_category": by_cat_out,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(sampled)} samples to {out_path}")
    return manifest


def prepare_3dsrbench_train(output_dir: Path, seed: int, n_samples: int = 300) -> dict:
    """3DSRBench: n_samples stratified from remainder (excluding frozen 500)."""
    print(f"\n[3DSRBench train {n_samples}] Loading...")
    ds = load_dataset("ccvl/3DSRBench", "benchmark", split="test")
    frozen_indices = get_frozen_3dsrbench_indices(ds, seed)
    remaining = [i for i in range(len(ds)) if i not in frozen_indices]
    print(f"  Full: {len(ds)}, Frozen: {len(frozen_indices)}, Remaining: {len(remaining)}")

    by_cat = {}
    pool = remaining if len(remaining) >= n_samples else list(range(len(ds)))
    if len(remaining) < n_samples:
        print(f"  [Warn] Remaining {len(remaining)} < {n_samples}, sampling from full (may overlap frozen)")
    for i in pool:
        c = ds[i].get("category") or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    rng = random.Random(seed + 1)
    indices = stratified_sample_from_pool(by_cat, n_samples, rng)
    sampled = ds.select(indices)

    out_path = output_dir / f"3dsrbench_train_{n_samples}"
    out_path.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(str(out_path))

    by_cat_out = {}
    for i in indices:
        c = ds[i].get("category") or "unknown"
        by_cat_out[c] = by_cat_out.get(c, 0) + 1
    manifest = {
        "benchmark": "3dsrbench",
        "split": "train",
        "n_samples": len(sampled),
        "excludes": "3dsrbench_500 (frozen)" if pool == remaining else "overlap (insufficient remaining)",
        "seed": seed,
        "source": "ccvl/3DSRBench",
        "subset": "benchmark",
        "per_category": by_cat_out,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(sampled)} samples to {out_path}")
    return manifest


def prepare_stvqa_train(output_dir: Path, seed: int, n_samples: int = 300) -> dict:
    """STVQA: n_samples stratified from train split (frozen uses val, no exclusion)."""
    print(f"\n[STVQA train {n_samples}] Loading...")
    ds = load_dataset("hunarbatra/STVQA-7K", split="train")
    print(f"  Train split: {len(ds)} samples")

    by_cat = {}
    for i in range(len(ds)):
        c = ds[i].get("category") or "unknown"
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(i)

    rng = random.Random(seed)
    indices = stratified_sample_from_pool(by_cat, n_samples, rng)
    sampled = ds.select(indices)

    out_path = output_dir / f"stvqa_train_{n_samples}"
    out_path.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(str(out_path))

    by_cat_out = {}
    for i in indices:
        c = ds[i].get("category") or "unknown"
        by_cat_out[c] = by_cat_out.get(c, 0) + 1
    manifest = {
        "benchmark": "stvqa",
        "split": "train",
        "n_samples": len(sampled),
        "source": "hunarbatra/STVQA-7K",
        "seed": seed,
        "per_category": by_cat_out,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {len(sampled)} samples to {out_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare train datasets (disjoint from frozen)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "dataset",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["cvbench", "3dsrbench", "stvqa"],
        choices=["cvbench", "3dsrbench", "stvqa"],
    )
    parser.add_argument("--n", "--n-samples", dest="n_samples", type=int, default=300,
                        help="Samples per dataset (stratified). Default 300. Use 500 for TTO comparison.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    readme = {
        "train_datasets": "Disjoint from frozen_benchmarks (used for training)",
        "seed": args.seed,
        "n_samples": n,
        "paths": {
            "cvbench": f"cvbench_train_{n}",
            "3dsrbench": f"3dsrbench_train_{n}",
            "stvqa": f"stvqa_train_{n}",
        },
    }
    (args.output_dir / "README.json").write_text(json.dumps(readme, indent=2))

    for key in args.benchmarks:
        if key == "cvbench":
            prepare_cvbench_train(args.output_dir, args.seed, n_samples=n)
        elif key == "3dsrbench":
            prepare_3dsrbench_train(args.output_dir, args.seed, n_samples=n)
        elif key == "stvqa":
            prepare_stvqa_train(args.output_dir, args.seed, n_samples=n)

    print("\nDone. Train datasets at", args.output_dir)


if __name__ == "__main__":
    main()
