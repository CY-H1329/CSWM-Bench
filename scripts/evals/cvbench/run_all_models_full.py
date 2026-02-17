#!/usr/bin/env python3
"""
Run all 3 GPU models (Qwen3, Sa2VA, LLaVA4D) on full CV-Bench.
Sequential execution. For with/without prompt, run separately.

Usage:
  python scripts/evals/cvbench/run_all_models_full.py
  python scripts/evals/cvbench/run_all_models_full.py --without_prompt
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = [
    "scripts/evals/cvbench/run_eval_cvbench_qwen3.py",
    "scripts/evals/cvbench/run_eval_cvbench_sa2va.py",
    "scripts/evals/cvbench/run_eval_cvbench_llava4d.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all 3 models on full CV-Bench")
    parser.add_argument("--without_prompt", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = ["--config", args.config, "--full_dataset", "--seed", str(args.seed)]
    if args.without_prompt:
        base.append("--without_prompt")

    for i, script in enumerate(SCRIPTS):
        path = ROOT / script
        if not path.exists():
            print(f"[!] Script not found: {path}")
            continue
        print("\n" + "=" * 60)
        print(f"[{i+1}/3] Running: {script}")
        print("=" * 60)
        ret = subprocess.run([sys.executable, str(path)] + base, cwd=str(ROOT))
        if ret.returncode != 0:
            print(f"[!] Failed: {script} (exit {ret.returncode})")
            sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("All 3 models completed on full CV-Bench.")
    print("=" * 60)


if __name__ == "__main__":
    main()
