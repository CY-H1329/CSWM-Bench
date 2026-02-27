#!/usr/bin/env python3
"""
Test full MAS v2 pipeline: Head → 3 Specialists → Final Reasoning Agent.

Measures Final Reasoning Agent accuracy on CV-Bench / 3DSRBench.
Uses prefetch for parallel image loading.

Usage (Jupyter on server):
    import sys
    sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")
    from test_final_reasoning_mas_v2 import run_mas_test
    from run_eval_mas_v2 import build_runners

    head_gen, spec_gen, reason_gen = build_runners(
        specialist_device="cuda",
        use_local_reasoning=True,  # API 없이 로컬 추론 (Jupyter 한 세션)
    )
    results = run_mas_test(
        head_gen, spec_gen, reason_gen,
        benchmark="cvbench",
        max_samples=100,
        prefetch_workers=4,
    )
    print(f"Accuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")

Usage (CLI):
    python test_final_reasoning_mas_v2.py --benchmark cvbench --max_samples 100
"""
import argparse
import random
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import (
    ALL_CATEGORIES,
    ScoreMap,
    run_step,
)
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_category,
)


def _prefetch_sample(ex, benchmark, i):
    """Prefetch image + metadata for one sample."""
    image = get_benchmark_image(ex, benchmark)
    if image is None:
        return None
    query = get_benchmark_prompt(ex, benchmark)
    gt_raw = get_benchmark_answer(ex, benchmark)
    category = get_benchmark_category(ex, benchmark) or "unknown"
    gt = (gt_raw or "").strip().upper()
    if not any(c in gt for c in "ABCD"):
        return None
    return {"index": i, "image": image, "query": query, "gt": gt, "category": category}


def run_mas_test(
    head_generate,
    specialist_generate,
    reasoning_generate,
    benchmark: str = "cvbench",
    max_samples: int = 100,
    seed: int = 42,
    prefetch_workers: int = 4,
    show_failures: int = 0,
    use_vlm_reasoning: bool = False,
):
    """
    Run full MAS v2 pipeline on benchmark.

    Returns dict with accuracy, correct, total, per_category, details.
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    score_map = ScoreMap(categories=ALL_CATEGORIES, seed=seed)

    # Prefetch samples
    if prefetch_workers > 0:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as exe:
            futures = {
                exe.submit(_prefetch_sample, dataset[i], benchmark, i): i
                for i in range(len(dataset))
            }
            prefetched = {}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    prefetched[r["index"]] = r
        samples = [prefetched[i] for i in sorted(prefetched.keys())]
    else:
        samples = []
        for i in range(len(dataset)):
            r = _prefetch_sample(dataset[i], benchmark, i)
            if r is not None:
                samples.append(r)

    # Shuffle to avoid dataset-ordering bias (e.g. easier samples clustered at end)
    rng = random.Random(seed)
    rng.shuffle(samples)

    reason_mode = "Qwen3-VL-8B (image+text)" if use_vlm_reasoning else "DeepSeek-R1 (text-only)"
    print(f"Starting: {len(samples)} samples, MAS v2 (Head + 3 Specialists + Final Reasoning [{reason_mode}])...")
    print(f"  prefetch_workers={prefetch_workers}")

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for step, s in enumerate(samples):
        result = run_step(
            image=s["image"],
            query=s["query"],
            gt=s["gt"],
            step=step + 1,
            total_steps=len(samples),
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=None,
            update_scores=False,
            use_vlm_reasoning=use_vlm_reasoning,
        )
        hit = result.get("correct", False)
        total += 1
        if hit:
            correct += 1
        cat = result.get("category", s["category"])
        by_category[cat]["total"] += 1
        if hit:
            by_category[cat]["correct"] += 1

        details.append({
            "index": step,
            "query": s["query"][:200],
            "gt": s["gt"],
            "final_answer": result.get("final_answer"),
            "correct": hit,
            "category": cat,
        })

        if (step + 1) % 10 == 0 or step == len(samples) - 1:
            acc = correct / total if total > 0 else 0
            print(f"  Progress {step+1}/{len(samples)} | acc: {100*acc:.1f}%")

    # Report
    print()
    print("=" * 60)
    print("MAS v2 — Final Reasoning Agent —", benchmark.upper())
    print("=" * 60)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print()
    for cat in sorted(by_category.keys()):
        v = by_category[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:35s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 60)

    failures = [d for d in details if not d["correct"]]
    if show_failures > 0 and failures:
        n = min(show_failures, len(failures))
        print()
        print("=" * 60)
        print(f"FAILURES (first {n})")
        print("=" * 60)
        for j, d in enumerate(failures[:n]):
            print(f"\n--- #{j+1} ---")
            print(f"Query: {d['query'][:300]}...")
            print(f"GT: {d['gt']}  |  Pred: {d['final_answer']}")
        print("=" * 60)

    return {
        "benchmark": benchmark,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_category": dict(by_category),
        "details": details,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="cvbench", choices=["cvbench", "3dsrbench"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch_workers", type=int, default=4)
    parser.add_argument("--show_failures", type=int, default=10)
    parser.add_argument("--reasoning_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--reasoning_api_key", type=str, default="EMPTY")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from run_eval_mas_v2 import build_runners

    head_gen, spec_gen, reason_gen = build_runners(
        reasoning_api_base=args.reasoning_api_base,
        reasoning_api_key=args.reasoning_api_key,
        specialist_device=args.device,
    )

    results = run_mas_test(
        head_gen,
        spec_gen,
        reason_gen,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        seed=args.seed,
        prefetch_workers=args.prefetch_workers,
        show_failures=args.show_failures,
    )

    print(f"\nSUMMARY: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
