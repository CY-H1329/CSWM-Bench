#!/usr/bin/env python3
"""
CV-Bench 1샘플로 MAS v2 지연 시간 테스트.

Optimized vs Legacy (object_extraction 공유 여부) 비교.

H100 서버 Jupyter에서:
    import sys
    sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")
    %run scripts/run_single_sample_latency_test.py

또는 CLI:
    python scripts/run_single_sample_latency_test.py
    python scripts/run_single_sample_latency_test.py --compare  # legacy vs optimized 둘 다
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src2.agents.mas_v2 import ALL_CATEGORIES, ScoreMap, run_step
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
)


def run_one(image, query, gt, head_gen, spec_gen, reason_gen, shared_extraction: bool):
    """Run one step with given shared_object_extraction setting."""
    score_map = ScoreMap(categories=ALL_CATEGORIES, seed=42)
    t0 = time.time()
    result = run_step(
        image=image,
        query=query,
        gt=gt,
        step=1,
        total_steps=1,
        score_map=score_map,
        head_generate=head_gen,
        specialist_generate=spec_gen,
        reasoning_generate=reason_gen,
        updater=None,
        update_scores=False,
        shared_object_extraction=shared_extraction,
    )
    elapsed = time.time() - t0
    return result, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run both legacy and optimized, compare times")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from run_eval_mas_v2 import build_runners

    print("Loading models...")
    head_gen, spec_gen, reason_gen = build_runners(
        specialist_device="cuda",
        use_local_reasoning=True,
        reasoning_local_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )

    dataset = load_benchmark("cvbench", max_samples=1, seed=args.seed)
    ex = dataset[0]
    image = get_benchmark_image(ex, "cvbench")
    query = get_benchmark_prompt(ex, "cvbench")
    gt = get_benchmark_answer(ex, "cvbench")

    if image is None:
        print("ERROR: Could not load image")
        return

    print("\n" + "=" * 60)
    print("CV-Bench 1샘플 MAS v2 지연 시간 테스트")
    print("=" * 60)
    print(f"Query: {query[:100]}...")
    print(f"GT: {gt}")
    print()

    if args.compare:
        # Legacy (object_extraction 2회)
        print("[1/2] Legacy (object_extraction per role, 2x)...")
        _, t_legacy = run_one(image, query, gt, head_gen, spec_gen, reason_gen, shared_extraction=False)
        print(f"      Legacy: {t_legacy:.1f} sec")

        # Optimized (object_extraction 1회)
        print("[2/2] Optimized (shared object_extraction, 1x)...")
        _, t_opt = run_one(image, query, gt, head_gen, spec_gen, reason_gen, shared_extraction=True)
        print(f"      Optimized: {t_opt:.1f} sec")

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Legacy:    {t_legacy:.1f} sec")
        print(f"Optimized: {t_opt:.1f} sec")
        print(f"Saved:     {t_legacy - t_opt:.1f} sec ({(1 - t_opt/t_legacy)*100:.0f}% faster)")
        print("=" * 60)
    else:
        # Optimized only
        print("Running (optimized, shared object_extraction)...")
        result, t = run_one(image, query, gt, head_gen, spec_gen, reason_gen, shared_extraction=True)
        print(f"\nElapsed: {t:.1f} sec")
        print(f"Final answer: {result.get('final_answer')}")
        print(f"Correct: {result.get('correct')}")

    print()


if __name__ == "__main__":
    main()
