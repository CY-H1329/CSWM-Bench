#!/usr/bin/env python3
"""
MAS v2 Baseline — Run all experiments (10, 50, 100 samples × 2 benchmarks).

Usage:
  cd Spatial_MAS
  python experiments/mas_v2_baseline/run_all.py

Or on H100:
  python experiments/mas_v2_baseline/run_all.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_eval_mas_v2 import build_runners, run_experiment

BENCHMARKS = ["cvbench", "3dsrbench"]
SAMPLE_SIZES = [10, 50, 100]
SEED = 42
OUTPUT_BASE = ROOT / "results" / "mas_v2_baseline"


def main():
    print("Building runners (use_local_reasoning=True for H100)...")
    head_gen, spec_gen, reason_gen = build_runners(
        specialist_device="cuda",
        use_local_reasoning=True,
        reasoning_local_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )

    all_results = []
    for benchmark in BENCHMARKS:
        for n in SAMPLE_SIZES:
            print(f"\n{'='*60}")
            print(f"{benchmark} | {n} samples")
            print("="*60)
            out_dir = str(OUTPUT_BASE / benchmark / f"{n}samples")
            out = run_experiment(
                benchmark=benchmark,
                head_generate=head_gen,
                specialist_generate=spec_gen,
                reasoning_generate=reason_gen,
                train_ratio=0.5,
                seed=SEED,
                output_dir=out_dir,
                max_samples=n,
            )
            all_results.append({
                "benchmark": benchmark,
                "samples": n,
                "train_acc": out["train_metrics"]["accuracy"],
                "test_acc": out["test_metrics"]["accuracy"],
                "train_n": out["train_metrics"]["total"],
                "test_n": out["test_metrics"]["total"],
            })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['benchmark']:12} | {r['samples']:3} samples | "
              f"train {r['train_acc']*100:.1f}% | test {r['test_acc']*100:.1f}%")
    print(f"\nResults saved to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
