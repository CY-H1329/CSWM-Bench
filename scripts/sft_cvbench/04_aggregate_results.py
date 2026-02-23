#!/usr/bin/env python3
"""
Aggregate SFT evaluation results into results_cvbench_scaling.csv.

Columns: Model, Shots per task, 2D Accuracy, 3D Accuracy, Overall Accuracy

Usage:
  python scripts/sft_cvbench/04_aggregate_results.py
  python scripts/sft_cvbench/04_aggregate_results.py --output results/sft_cvbench/results_cvbench_scaling.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

try:
    import pandas as pd
except ImportError:
    pd = None


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate SFT results")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="results_cvbench_scaling.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config_sft.yaml"
    config = load_config(config_path)

    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = config["models"]
    shots_list = config["shots_per_task"]
    split_name = "human_test"

    rows = []
    for model in models:
        for shots in shots_list:
            results_path = output_dir / model / str(shots) / split_name / "results.json"
            if not results_path.exists():
                continue
            with open(results_path) as f:
                r = json.load(f)
            rows.append({
                "Model": model,
                "Shots per task": shots,
                "2D Accuracy": r.get("accuracy_2d", 0.0),
                "3D Accuracy": r.get("accuracy_3d", 0.0),
                "Overall Accuracy": r.get("overall_accuracy", 0.0),
            })

    if not rows:
        print("No results found. Run 03_evaluate.py for each model/shots first.")
        sys.exit(0)

    out_csv = output_dir / args.output_file
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(df.to_string(index=False))
    else:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Model", "Shots per task", "2D Accuracy", "3D Accuracy", "Overall Accuracy"])
            w.writeheader()
            w.writerows(rows)
        for row in rows:
            print(row)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
