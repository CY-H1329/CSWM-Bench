#!/usr/bin/env python3
"""
Exécute les 3 modèles (Qwen3, Sa2VA, LLaVA4D) sur le dataset 3DSRBench complet.
Chaque modèle s'exécute dans un processus séparé pour éviter les fuites mémoire.

Usage:
  python scripts/evals/3dsrbench/run_all_models_full.py
  python scripts/evals/3dsrbench/run_all_models_full.py --seed 42
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = [
    "scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py",
    "scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py",
    "scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all 3 models on full 3DSRBench")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    base = ["--config", args.config, "--seed", str(args.seed), "--max_new_tokens", str(args.max_new_tokens)]
    # Pas de --max_samples = dataset complet

    for i, script in enumerate(SCRIPTS):
        path = ROOT / script
        if not path.exists():
            print(f"[!] Script not found: {path}")
            continue
        print("\n" + "=" * 60)
        print(f"[{i+1}/3] Running: {script}")
        print("=" * 60)
        ret = subprocess.run(
            [sys.executable, str(path)] + base,
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            print(f"[!] Failed: {script} (exit {ret.returncode})")
            sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("All 3 models completed on full dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
