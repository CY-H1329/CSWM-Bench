#!/usr/bin/env python3
"""
Head Agent classification test — 5 unified categories (A-approach).

Strategy: Head Agent ALWAYS receives all 16 fine-grained categories
regardless of benchmark. Model classifies into one of 16, then we map
to 5 unified categories post-hoc via FINE_TO_UNIFIED.

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
    ALL_CATEGORIES, FINE_TO_UNIFIED,
    ALL_FINE_CATEGORIES, FINE_CATEGORY_DESCRIPTIONS,
)
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.benchmarks.loaders import (
    load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_category,
)


def parse_fine_category(raw):
    """Parse model output into one of the 16 fine-grained categories."""
    raw_clean = (raw or "").strip().lower()
    for cat in ALL_FINE_CATEGORIES:
        if cat.lower() == raw_clean:
            return cat
    for cat in ALL_FINE_CATEGORIES:
        if cat.lower() in raw_clean:
            return cat
    return "UNKNOWN"


def run_5cat_test(runner, benchmark="cvbench", max_samples=None, seed=42):
    """Run Head Agent 5-category classification test (A-approach).

    The model ALWAYS sees all 16 fine-grained categories, then we map
    to 5 unified categories post-hoc via FINE_TO_UNIFIED.

    Args:
        runner: Qwen3Runner instance (must have .generate(image, prompt) method)
        benchmark: "cvbench" or "3dsrbench"
        max_samples: limit samples (None = all)
        seed: random seed
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)

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

        if image is None or raw_gt not in FINE_TO_UNIFIED:
            continue

        prompt = build_head_agent_prompt(
            query, ALL_FINE_CATEGORIES, FINE_CATEGORY_DESCRIPTIONS,
        )
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred_fine = parse_fine_category(raw)

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
    active_fine = [c for c in ALL_FINE_CATEGORIES if c in fine_by_gt]
    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — FINE-GRAINED (16 categories, {len(active_fine)} active)")
    print("=" * 70)
    print(f"Overall: {fine_correct}/{total} = {100*fine_correct/total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN: {unknown_count}")
    print()
    for cat in active_fine:
        v = fine_by_gt[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            grp = FINE_TO_UNIFIED.get(cat, "?")
            print(f"  [{grp:18s}] {cat:45s} {100*acc:5.1f}%  ({v['correct']}/{v['total']})")

    # --- Unified report ---
    active_unified = [c for c in ALL_CATEGORIES if c in unified_by_gt]
    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — UNIFIED (5 categories)")
    print("=" * 70)
    print(f"Overall: {unified_correct}/{total} = {100*unified_correct/total:.1f}%")
    print()

    for cat in active_unified:
        v = unified_by_gt[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            members = [f for f in active_fine if FINE_TO_UNIFIED.get(f) == cat]
            print(f"  {cat:18s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
            print(f"    members: {', '.join(members)}")
            print()

    all_pred = active_unified + (["UNKNOWN"] if unknown_count else [])
    print("Unified confusion (row=GT, col=Pred):")
    header = f"{'':18s}" + "".join(f"{c:>18s}" for c in all_pred)
    print(header)
    print("-" * len(header))
    for gt_cat in active_unified:
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
