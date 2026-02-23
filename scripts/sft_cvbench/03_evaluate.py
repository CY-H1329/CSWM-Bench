#!/usr/bin/env python3
"""
SFT Evaluation script for CV-Bench.

Evaluates a trained model on human_selected_test_set.
Computes: 2D accuracy, 3D accuracy, overall accuracy.

Usage:
  python scripts/sft_cvbench/03_evaluate.py --model qwen3_4b --shots 10 --checkpoint path/to/ckpt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# 2D = Count, Relation | 3D = Depth, Distance
CVBENCH_2D = {"Count", "Relation"}
CVBENCH_3D = {"Depth", "Distance"}


def parse_args():
    parser = argparse.ArgumentParser(description="SFT evaluation on CV-Bench")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen3_4b", "llava4d", "sa2va", "spatialrgpt", "spatialreasoner"],
    )
    parser.add_argument("--shots", type=int, required=True, choices=[10, 30, 100])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="human_test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit for debug")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config_sft.yaml"
    config = load_config(config_path)

    splits_dir = Path(config["paths"]["splits_dir"])
    split_path = splits_dir / f"{args.split}.json"
    if not split_path.exists():
        print(f"ERROR: Missing split: {split_path}. Run 01_sample_dataset.py first.")
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)
    indices = split_data["indices"]
    if args.max_samples:
        indices = indices[: args.max_samples]

    out_dir = args.output_dir or Path(config["paths"]["output_dir"]) / args.model / str(args.shots) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SFT Evaluation (CV-Bench)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Shots: {args.shots}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split} ({len(indices)} samples)")
    print(f"Output: {out_dir}")
    print()

    # Placeholder: actual inference uses model runner + get_benchmark_*
    # Results format for 04_aggregate_results:
    results = {
        "model": args.model,
        "shots": args.shots,
        "split": args.split,
        "n_samples": len(indices),
        "overall_accuracy": 0.0,
        "accuracy_2d": 0.0,
        "accuracy_3d": 0.0,
        "task_accuracy": {"Count": 0.0, "Relation": 0.0, "Depth": 0.0, "Distance": 0.0},
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("NOTE: Full evaluation requires model inference.")
    print("  Integrate with src/models/ runners and src/benchmarks loaders.")
    print(f"  Placeholder results saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
