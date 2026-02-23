#!/usr/bin/env python3
"""
Head Agent classification test — 5 unified categories.

Grounded in cognitive neuroscience (Kosslyn 1987, Walsh 2003, Levinson 2003):
  1. spatial_relation  — WHERE?    (categorical spatial processing)
  2. distance_depth    — HOW FAR?  (coordinate spatial processing)
  3. size              — HOW BIG?  (magnitude processing)
  4. orientation       — WHICH WAY? (mental rotation / direction)
  5. counting          — HOW MANY? (numerosity processing)

Usage (Jupyter):
    from test_head_5cat_unified import run_5cat_test
    results = run_5cat_test(runner, benchmark="cvbench", max_samples=400)
    results = run_5cat_test(runner, benchmark="3dsrbench", max_samples=200)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2.config import (
    ALL_CATEGORIES, CATEGORY_DESCRIPTIONS, FINE_TO_UNIFIED,
)
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.benchmarks.loaders import (
    load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_category,
)


def parse_category_5(raw):
    """Parse model output into one of the 5 unified categories."""
    raw_clean = (raw or "").strip().lower()
    for cat in ALL_CATEGORIES:
        if cat == raw_clean:
            return cat
    for cat in ALL_CATEGORIES:
        if cat in raw_clean:
            return cat
    # Fallback: try to match known fine-grained names → unified
    for fine, unified in FINE_TO_UNIFIED.items():
        if fine.lower() in raw_clean:
            return unified
    return "UNKNOWN"


def normalize_gt(raw_gt):
    """Map benchmark GT category to unified 5-category name."""
    if not raw_gt:
        return None
    raw_gt = raw_gt.strip()
    if raw_gt in FINE_TO_UNIFIED:
        return FINE_TO_UNIFIED[raw_gt]
    if raw_gt.lower() in {c.lower() for c in ALL_CATEGORIES}:
        for c in ALL_CATEGORIES:
            if c.lower() == raw_gt.lower():
                return c
    return None


def run_5cat_test(runner, benchmark="cvbench", max_samples=None, seed=42):
    """Run Head Agent 5-category classification test.

    Args:
        runner: Qwen3Runner instance (must have .generate(image, prompt) method)
        benchmark: "cvbench" or "3dsrbench"
        max_samples: limit samples (None = all)
        seed: random seed
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)

    correct = 0
    total = 0
    unknown_count = 0
    by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, benchmark)
        query = get_benchmark_prompt(ex, benchmark)
        raw_gt = get_benchmark_category(ex, benchmark)
        gt = normalize_gt(raw_gt)

        if image is None or gt is None:
            continue

        prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred = parse_category_5(raw)

        if pred == "UNKNOWN":
            unknown_count += 1

        total += 1
        hit = pred == gt
        if hit:
            correct += 1
        by_gt[gt]["total"] += 1
        if hit:
            by_gt[gt]["correct"] += 1
        confusion[gt][pred] += 1
        details.append({
            "index": i, "gt": gt, "gt_raw": raw_gt,
            "pred": pred, "raw": raw.strip()[:80],
            "correct": hit,
        })

        if (i + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  Progress {i+1}/{len(dataset)} | evaluated: {total} | acc: {100*acc:.1f}%")

    if total == 0:
        print("No samples evaluated!")
        return {}

    # --- Report ---
    print()
    print("=" * 60)
    title = f"HEAD AGENT — 5 unified categories — {benchmark.upper()}"
    print(title)
    print("=" * 60)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN outputs: {unknown_count}")
    print()

    for cat in ALL_CATEGORIES:
        v = by_gt.get(cat)
        if v and v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:20s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")

    all_pred_cats = ALL_CATEGORIES + (["UNKNOWN"] if unknown_count else [])
    print()
    print("Confusion (row=GT, col=Pred):")
    header = f"{'':20s}" + "".join(f"{c:>16s}" for c in all_pred_cats)
    print(header)
    print("-" * len(header))
    for gt_cat in ALL_CATEGORIES:
        if gt_cat not in confusion:
            continue
        row = f"{gt_cat:20s}"
        for pred_cat in all_pred_cats:
            row += f"{confusion[gt_cat].get(pred_cat, 0):>16d}"
        print(row)
    print("=" * 60)

    return {
        "benchmark": benchmark,
        "overall_accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "unknown": unknown_count,
        "per_category": {
            cat: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                **v,
            }
            for cat, v in by_gt.items()
        },
        "confusion": dict(confusion),
        "details": details,
    }
