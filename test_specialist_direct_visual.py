#!/usr/bin/env python3
"""
Test single specialist agent: direct_visual_heuristic + Qwen3-VL-4B.

Measures accuracy (predicted answer vs GT) on CV-Bench and/or 3DSRBench.

Usage (Jupyter):
    from src2.models.qwen3 import Qwen3Runner
    runner = Qwen3Runner(device="cuda")

    from test_specialist_direct_visual import run_specialist_test
    results = run_specialist_test(runner, benchmark="cvbench", max_samples=100)
    results = run_specialist_test(runner, benchmark="3dsrbench", max_samples=100)

Usage (CLI):
    python test_specialist_direct_visual.py --benchmark cvbench --max_samples 100
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2.prompts import build_role_prompt
from src2.agents.mas_v2.pipeline import parse_specialist_output
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_category,
)


def _normalize_answer(s: str) -> str:
    """Extract answer letter A/B/C/D for comparison."""
    s = (s or "").strip().upper()
    m = re.search(r"\(?([A-D])\)?", s)
    if m:
        return m.group(1)
    if s in "ABCD":
        return s
    return ""


def run_specialist_test(
    runner,
    benchmark: str = "cvbench",
    max_samples: int = None,
    seed: int = 42,
    show_failures: int = 0,
):
    """
    Run direct_visual_heuristic + Qwen3 on benchmark, report accuracy vs GT.

    Args:
        runner: Must have .generate(image, prompt, temperature=0, max_new_tokens=N)
        benchmark: "cvbench" or "3dsrbench"
        max_samples: Limit samples (None = all)
        seed: Random seed
        show_failures: Print first N wrong cases (query, GT, pred, reason)
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    role = "direct_visual_heuristic"

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, benchmark)
        query = get_benchmark_prompt(ex, benchmark)
        gt_raw = get_benchmark_answer(ex, benchmark)
        category = get_benchmark_category(ex, benchmark) or "unknown"

        if image is None:
            continue

        gt_letter = _normalize_answer(gt_raw)
        if not gt_letter:
            continue

        prompt = build_role_prompt(role, query, tool_output=None)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=512)
        answer, reason = parse_specialist_output(raw)
        pred_letter = _normalize_answer(answer)

        total += 1
        hit = pred_letter == gt_letter
        if hit:
            correct += 1
        by_category[category]["total"] += 1
        if hit:
            by_category[category]["correct"] += 1

        details.append({
            "index": i,
            "query": query,
            "gt": gt_letter,
            "pred": pred_letter,
            "correct": hit,
            "category": category,
            "raw_answer": answer,
            "reason": reason,
            "raw_output": raw[:3000] if raw else "",
        })

        if (i + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  Progress {i+1}/{len(dataset)} | acc: {100*acc:.1f}%")

    if total == 0:
        print("No samples evaluated!")
        return {}

    # --- Report ---
    print()
    print("=" * 60)
    print(f"SPECIALIST TEST — {role} + Qwen3-VL-4B — {benchmark.upper()}")
    print("=" * 60)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print()

    for cat in sorted(by_category.keys()):
        v = by_category[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:30s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 60)

    failures = [d for d in details if not d["correct"]]

    # --- Show first N failures ---
    if show_failures > 0 and failures:
        n = min(show_failures, len(failures))
        print()
        print("=" * 60)
        print(f"FAILURE ANALYSIS (first {n} of {len(failures)} wrong)")
        print("=" * 60)
        for j, d in enumerate(failures[:n]):
            print()
            print(f"--- Failure #{j+1} (index={d['index']}, category={d['category']}) ---")
            print(f"Query:\n{d['query'][:500]}{'...' if len(d['query']) > 500 else ''}")
            print(f"GT: {d['gt']}  |  Pred: {d['pred']}")
            reason_preview = (d.get("reason") or "")[:600]
            if reason_preview:
                print(f"Model's Reason (truncated):\n{reason_preview}...")
            print("-" * 40)
        print("=" * 60)

    return {
        "benchmark": benchmark,
        "role": role,
        "model": "qwen3_4b",
        "accuracy": correct / total,
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
    parser.add_argument("--show_failures", type=int, default=0, help="Print first N wrong cases")
    args = parser.parse_args()

    from src2.models.qwen3 import Qwen3Runner
    runner = Qwen3Runner(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    run_specialist_test(
        runner,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        seed=args.seed,
        show_failures=args.show_failures,
    )
