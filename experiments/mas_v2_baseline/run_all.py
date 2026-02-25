#!/usr/bin/env python3
"""
MAS v2 Baseline — Testing only (no train/test split).

Pipeline: Head → ScoreMap (random) → 3 Specialists → SharedMemory → Final Reasoning
Benchmarks: CV-Bench, 3DSRBench
Sample sizes: 10, 50, 100

Usage:
  cd Spatial_MAS
  python experiments/mas_v2_baseline/run_all.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_eval_mas_v2 import build_runners, run_test_only

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
            print(f"{benchmark} | {n} samples (testing only)")
            print("="*60)
            out_dir = str(OUTPUT_BASE / benchmark / f"{n}samples")
            out = run_test_only(
                benchmark=benchmark,
                head_generate=head_gen,
                specialist_generate=spec_gen,
                reasoning_generate=reason_gen,
                max_samples=n,
                seed=SEED,
                output_dir=out_dir,
            )
            m = out["metrics"]
            all_results.append({
                "benchmark": benchmark,
                "samples": n,
                "accuracy": m["accuracy"],
                "correct": m["correct"],
                "total": m["total"],
            })

    print("\n" + "="*60)
    print("SUMMARY (testing only, random agents)")
    print("="*60)
    for r in all_results:
        print(f"  {r['benchmark']:12} | {r['samples']:3} samples | "
              f"accuracy {r['accuracy']*100:.1f}% ({r['correct']}/{r['total']})")
    print(f"\nResults saved to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
