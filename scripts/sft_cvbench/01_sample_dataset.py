#!/usr/bin/env python3
"""
CV-Bench SFT Dataset Sampler.

Creates:
- human_selected_test_set: 300 2D (Count+Relation) + 300 3D (Depth+Distance), stratified
- train splits: for each shots_per_task, stratified sample per task
- Saves indices to data/sft_cvbench/splits/ as JSON
- Fixed seed for reproducibility. Same samples for all models.
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from datasets import load_dataset

# CV-Bench categories: 2D = Count, Relation | 3D = Depth, Distance
CVBENCH_2D_TASKS = ["Count", "Relation"]
CVBENCH_3D_TASKS = ["Depth", "Distance"]
CVBENCH_ALL_TASKS = CVBENCH_2D_TASKS + CVBENCH_3D_TASKS


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratify_by_task(dataset, category_key: str = "task") -> Dict[str, List[int]]:
    """Group indices by task. Returns {task: [indices]}."""
    by_task = defaultdict(list)
    for i in range(len(dataset)):
        task = (dataset[i].get(category_key) or "").strip()
        if task in CVBENCH_ALL_TASKS:
            by_task[task].append(i)
    return dict(by_task)


def sample_stratified(
    by_task: Dict[str, List[int]],
    n_per_task: int,
    rng: random.Random,
    exclude_indices: Set[int] = None,
) -> List[int]:
    """Sample n_per_task from each task, excluding exclude_indices."""
    if exclude_indices is None:
        exclude_indices = set()
    sampled = []
    for task, indices in by_task.items():
        available = [idx for idx in indices if idx not in exclude_indices]
        if not available:
            continue
        n = min(n_per_task, len(available))
        sampled.extend(rng.sample(available, n))
    return sorted(sampled)


def main():
    config = load_config()
    seed = config["seed"]
    shots_per_task = config["shots_per_task"]
    n_test_2d = config["human_test_2d"]
    n_test_3d = config["human_test_3d"]
    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CV-Bench SFT Dataset Sampler")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Shots per task: {shots_per_task}")
    print(f"Test 2D: {n_test_2d}, Test 3D: {n_test_3d}")
    print()

    rng = random.Random(seed)

    print("Loading CV-Bench...")
    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    print(f"  Loaded: {len(ds)} samples")

    by_task = stratify_by_task(ds)
    for t in CVBENCH_ALL_TASKS:
        print(f"  {t}: {len(by_task.get(t, []))}")

    # Human test set: 300 2D + 300 3D (stratified)
    n_per_2d = n_test_2d // len(CVBENCH_2D_TASKS)
    n_per_3d = n_test_3d // len(CVBENCH_3D_TASKS)
    test_2d = sample_stratified(
        {k: by_task[k] for k in CVBENCH_2D_TASKS if k in by_task}, n_per_2d, rng
    )
    test_3d = sample_stratified(
        {k: by_task[k] for k in CVBENCH_3D_TASKS if k in by_task}, n_per_3d, rng
    )
    test_indices = sorted(test_2d + test_3d)
    test_set = set(test_indices)

    print(f"\nHuman test set: {len(test_indices)} samples")

    all_splits = {"human_test": test_indices}

    for shots in shots_per_task:
        train_indices = sample_stratified(by_task, shots, rng, exclude_indices=test_set)
        all_splits[f"train_{shots}"] = train_indices
        print(f"  train_{shots}: {len(train_indices)} samples")

    for name, indices in all_splits.items():
        out_path = splits_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump({"indices": indices, "size": len(indices)}, f, indent=2)
        print(f"  Saved: {out_path}")

    summary = {name: len(indices) for name, indices in all_splits.items()}
    with open(splits_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
