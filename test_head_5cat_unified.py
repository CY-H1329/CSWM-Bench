#!/usr/bin/env python3
"""
Head Agent classification test — 5 unified categories (A-approach).

Strategy: Head Agent classifies into fine-grained benchmark categories
(12 for 3DSRBench, 4 for CV-Bench), then we map to 5 unified categories
post-hoc via FINE_TO_UNIFIED. This preserves the 98.5% grouped accuracy.

Unified categories (neuroscience-backed):
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
    CATEGORIES_3DSRBENCH, CATEGORIES_CVBENCH, FINE_CATEGORY_DESCRIPTIONS,
)
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.benchmarks.loaders import (
    load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_category,
)


def _get_fine_categories(benchmark):
    if benchmark == "3dsrbench":
        return CATEGORIES_3DSRBENCH
    elif benchmark == "cvbench":
        return CATEGORIES_CVBENCH
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def parse_fine_category(raw, fine_cats):
    """Parse model output into one of the fine-grained categories."""
    raw_clean = (raw or "").strip().lower()
    for cat in fine_cats:
        if cat.lower() == raw_clean:
            return cat
    for cat in fine_cats:
        if cat.lower() in raw_clean:
            return cat
    return "UNKNOWN"


def run_5cat_test(runner, benchmark="cvbench", max_samples=None, seed=42):
    """Run Head Agent 5-category classification test (A-approach).

    The model classifies into fine-grained categories, then we map to
    5 unified categories post-hoc via FINE_TO_UNIFIED.

    Args:
        runner: Qwen3Runner instance (must have .generate(image, prompt) method)
        benchmark: "cvbench" or "3dsrbench"
        max_samples: limit samples (None = all)
        seed: random seed
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    fine_cats = _get_fine_categories(benchmark)

    fine_correct = 0
    unified_correct = 0
    total = 0
    unknown_count = 0

    fine_by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    unified_by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    unified_confusion = defaultdict(lambda: defaultdict(int))
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, benchmark)
        query = get_benchmark_prompt(ex, benchmark)
        raw_gt = (get_benchmark_category(ex, benchmark) or "").strip()

        if image is None or raw_gt not in fine_cats:
            continue

        prompt = build_head_agent_prompt(query, fine_cats, FINE_CATEGORY_DESCRIPTIONS)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred_fine = parse_fine_category(raw, fine_cats)

        if pred_fine == "UNKNOWN":
            unknown_count += 1

        total += 1

        gt_unified = FINE_TO_UNIFIED.get(raw_gt, "UNKNOWN")
        pred_unified = FINE_TO_UNIFIED.get(pred_fine, "UNKNOWN")

        fine_hit = pred_fine == raw_gt
        unified_hit = gt_unified == pred_unified
        if fine_hit:
            fine_correct += 1
        if unified_hit:
            unified_correct += 1

        fine_by_gt[raw_gt]["total"] += 1
        if fine_hit:
            fine_by_gt[raw_gt]["correct"] += 1

        unified_by_gt[gt_unified]["total"] += 1
        if unified_hit:
            unified_by_gt[gt_unified]["correct"] += 1
        unified_confusion[gt_unified][pred_unified] += 1

        details.append({
            "index": i, "gt_fine": raw_gt, "pred_fine": pred_fine,
            "gt_unified": gt_unified, "pred_unified": pred_unified,
            "fine_correct": fine_hit, "unified_correct": unified_hit,
            "raw": raw.strip()[:80],
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress {i+1}/{len(dataset)} "
                  f"| fine: {100*fine_correct/total:.1f}% "
                  f"| unified: {100*unified_correct/total:.1f}%")

    if total == 0:
        print("No samples evaluated!")
        return {}

    # --- Fine-grained report ---
    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — FINE-GRAINED ({len(fine_cats)} categories)")
    print("=" * 70)
    print(f"Overall: {fine_correct}/{total} = {100*fine_correct/total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN: {unknown_count}")
    print()
    for cat in fine_cats:
        v = fine_by_gt.get(cat)
        if v and v["total"] > 0:
            acc = v["correct"] / v["total"]
            grp = FINE_TO_UNIFIED.get(cat, "?")
            print(f"  [{grp:18s}] {cat:45s} {100*acc:5.1f}%  ({v['correct']}/{v['total']})")

    # --- Unified report ---
    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — UNIFIED (5 categories)")
    print("=" * 70)
    print(f"Overall: {unified_correct}/{total} = {100*unified_correct/total:.1f}%")
    print()

    active_cats = [c for c in ALL_CATEGORIES if c in unified_by_gt]
    for cat in active_cats:
        v = unified_by_gt[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            members = [f for f, u in FINE_TO_UNIFIED.items() if u == cat and f in fine_cats]
            print(f"  {cat:18s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
            print(f"    members: {', '.join(members)}")
            print()

    all_pred = active_cats + (["UNKNOWN"] if unknown_count else [])
    print("Unified confusion (row=GT, col=Pred):")
    header = f"{'':18s}" + "".join(f"{c:>18s}" for c in all_pred)
    print(header)
    print("-" * len(header))
    for gt_cat in active_cats:
        row = f"{gt_cat:18s}"
        for pred_cat in all_pred:
            row += f"{unified_confusion[gt_cat].get(pred_cat, 0):>18d}"
        print(row)
    print("=" * 70)

    return {
        "benchmark": benchmark,
        "fine_accuracy": fine_correct / total,
        "unified_accuracy": unified_correct / total,
        "fine_correct": fine_correct,
        "unified_correct": unified_correct,
        "total": total,
        "unknown": unknown_count,
        "fine_per_category": dict(fine_by_gt),
        "unified_per_category": dict(unified_by_gt),
        "unified_confusion": dict(unified_confusion),
        "details": details,
    }
