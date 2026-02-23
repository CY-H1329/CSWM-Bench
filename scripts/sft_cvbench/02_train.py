#!/usr/bin/env python3
"""
SFT Training script for CV-Bench.

Trains a VLM on stratified CV-Bench samples.
Checkpoint saved as: {model_name}_cvbench_{shots}shot/

Usage:
  python scripts/sft_cvbench/02_train.py --model qwen3_4b --shots 10
  python scripts/sft_cvbench/02_train.py --model qwen3_4b --shots 30 --config scripts/sft_cvbench/config_sft.yaml
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import yaml


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training on CV-Bench")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen3_4b", "llava4d", "sa2va", "spatialrgpt", "spatialreasoner"],
    )
    parser.add_argument("--shots", type=int, required=True, choices=[10, 30, 100])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Limit data for quick test")
    parser.add_argument("--spatial_prompt", action="store_true", help="Use full spatial prompt (longer)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config_sft.yaml"
    config = load_config(config_path)

    splits_dir = Path(config["paths"]["splits_dir"])
    train_path = splits_dir / f"train_{args.shots}.json"
    if not train_path.exists():
        print(f"ERROR: Run 01_sample_dataset.py first. Missing: {train_path}")
        sys.exit(1)

    with open(train_path) as f:
        split_data = json.load(f)
    train_indices = split_data["indices"]
    if args.debug:
        train_indices = train_indices[:20]
        print(f"[DEBUG] Using {len(train_indices)} samples")

    out_dir = args.output_dir or Path(config["paths"]["checkpoints_dir"]) / f"{args.model}_cvbench_{args.shots}shot"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]

    print("=" * 70)
    print("SFT Training (CV-Bench)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Shots: {args.shots}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Output: {out_dir}")
    print()

    if args.model == "qwen3_4b":
        from train_impl.train_qwen3 import train_qwen3
        train_qwen3(
            train_indices=train_indices,
            output_dir=str(out_dir),
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            epochs=train_cfg.get("epochs", 3),
            batch_size=train_cfg.get("batch_size", 4),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            max_length=2048,
            use_spatial_prompt=args.spatial_prompt,
        )
    else:
        run_config = {
            "model": args.model,
            "shots": args.shots,
            "train_indices": train_indices[:5],
            "training": train_cfg,
        }
        with open(out_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)
        print(f"Model {args.model}: use LLaMA-Factory or add train_impl/train_{args.model}.py")
        print(f"  Config saved to {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
