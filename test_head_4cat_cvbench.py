#!/usr/bin/env python3
"""
Head Agent classification test — CV-Bench 4 categories only.

Tests whether the Head Agent can correctly classify CV-Bench questions
when given ONLY the 4 CV-Bench categories (count, relation, depth, distance),
without the 12 fine-grained 3DSRBench categories that cause overlap.

Usage (Jupyter):
    from test_head_4cat_cvbench import run_4cat_test
    results = run_4cat_test(runner, max_samples=100)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2.config import CATEGORY_DESCRIPTIONS
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.agents.mas_v2.pipeline import parse_category
from src2.benchmarks.loaders import (
    load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_category,
)

CVBENCH_4 = ["count", "relation", "depth", "distance"]
GT_MAP = {"Count": "count", "Relation": "relation", "Depth": "depth", "Distance": "distance"}

# 16-category → 4-category fallback mapping.
# When the model outputs a fine-grained category not in CVBENCH_4,
# map it to the corresponding coarse category instead of defaulting to [0].
FINE_TO_COARSE = {
    "location_above": "relation",
    "height_higher": "relation",
    "location_closer_to_camera": "depth",
    "multi_object_closer_to": "distance",
    "location_next_to": "relation",
    "orientation_on_the_left": "relation",
    "orientation_in_front_of": "relation",
    "orientation_viewpoint": "relation",
    "multi_object_facing": "relation",
    "multi_object_same_direction": "relation",
    "multi_object_viewpoint_towards_object": "depth",
    "multi_object_parallel": "relation",
}


def parse_category_4(raw: str) -> str:
    """Parse model output into one of the 4 CV-Bench categories."""
    raw_clean = (raw or "").strip().lower()
    # 1. Exact match to 4 categories
    for cat in CVBENCH_4:
        if cat == raw_clean:
            return cat
    # 2. Substring match to 4 categories
    for cat in CVBENCH_4:
        if cat in raw_clean:
            return cat
    # 3. Fine-grained 16-category → coarse 4-category mapping
    for fine, coarse in FINE_TO_COARSE.items():
        if fine in raw_clean:
            return coarse
    # 4. Unknown output
    return "UNKNOWN"


def run_4cat_test(runner, max_samples=None, seed=42):
    dataset = load_benchmark("cvbench", max_samples=max_samples, seed=seed)
    correct = 0
    total = 0
    unknown_count = 0
    by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = GT_MAP.get(get_benchmark_category(ex, "cvbench"))
        if image is None or gt is None:
            continue

        prompt = build_head_agent_prompt(query, CVBENCH_4, CATEGORY_DESCRIPTIONS)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred = parse_category_4(raw)
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
        details.append({"index": i, "gt": gt, "pred": pred, "raw": raw.strip()[:80], "correct": hit})

        if (i + 1) % 50 == 0:
            print(f"  Progress {i+1}/{len(dataset)} | acc: {100*correct/total:.1f}%")

    # --- Print report ---
    print()
    print("=" * 55)
    print("HEAD AGENT — CV-Bench 4 categories (with fine→coarse mapping)")
    print("=" * 55)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN outputs (model answered instead of classifying): {unknown_count}")
    print()
    for cat in sorted(by_gt):
        v = by_gt[cat]
        acc = v["correct"] / v["total"] if v["total"] > 0 else 0
        print(f"  {cat:15s}  {100*acc:.1f}%  ({v['correct']}/{v['total']})")

    print()
    all_pred_cats = CVBENCH_4 + (["UNKNOWN"] if unknown_count else [])
    print("Confusion (row=GT, col=Pred):")
    header = f"{'':15s}" + "".join(f"  {c:>10s}" for c in all_pred_cats)
    print(header)
    print("-" * len(header))
    for gt_cat in CVBENCH_4:
        row = f"{gt_cat:15s}"
        for pred_cat in all_pred_cats:
            row += f"  {confusion[gt_cat][pred_cat]:>10d}"
        print(row)
    print("=" * 55)

    return {
        "overall_accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_category": {
            cat: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0, **v}
            for cat, v in by_gt.items()
        },
        "confusion": dict(confusion),
        "details": details,
    }
