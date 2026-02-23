#!/usr/bin/env python3
"""
Head Agent classification test — 3DSRBench 12 categories.

Usage (Jupyter):
    from test_head_12cat_3dsrbench import run_12cat_test
    results = run_12cat_test(runner, max_samples=200)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2.config import CATEGORY_DESCRIPTIONS
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.benchmarks.loaders import (
    load_benchmark, get_benchmark_image, get_benchmark_prompt, get_benchmark_category,
)

CATS_12 = [
    "location_above",
    "height_higher",
    "location_closer_to_camera",
    "multi_object_closer_to",
    "location_next_to",
    "orientation_on_the_left",
    "orientation_in_front_of",
    "orientation_viewpoint",
    "multi_object_facing",
    "multi_object_same_direction",
    "multi_object_viewpoint_towards_object",
    "multi_object_parallel",
]


def parse_category_12(raw):
    raw_clean = (raw or "").strip().lower()
    for cat in CATS_12:
        if cat == raw_clean:
            return cat
    for cat in CATS_12:
        if cat in raw_clean:
            return cat
    return "UNKNOWN"


def run_12cat_test(runner, max_samples=None, seed=42):
    dataset = load_benchmark("3dsrbench", max_samples=max_samples, seed=seed)
    correct = 0
    total = 0
    unknown_count = 0
    by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, "3dsrbench")
        query = get_benchmark_prompt(ex, "3dsrbench")
        raw_gt = get_benchmark_category(ex, "3dsrbench")
        gt = (raw_gt or "").strip()
        if image is None or gt not in CATS_12:
            if image is None:
                print(f"  [skip] sample {i}: image fetch failed")
            continue

        prompt = build_head_agent_prompt(query, CATS_12, CATEGORY_DESCRIPTIONS)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred = parse_category_12(raw)
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
            "index": i, "gt": gt, "pred": pred,
            "raw": raw.strip()[:80], "correct": hit,
            "query": query[:150],
        })

        if (i + 1) % 50 == 0:
            evaluated = len(details)
            acc = correct / evaluated if evaluated > 0 else 0
            print(f"  Progress {i+1}/{len(dataset)} | evaluated: {evaluated} | acc: {100*acc:.1f}%")

    if total == 0:
        print("No samples evaluated!")
        return {}

    # --- Print report ---
    print()
    print("=" * 65)
    print("HEAD AGENT — 3DSRBench 12 categories")
    print("=" * 65)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN outputs: {unknown_count}")
    print()
    for cat in CATS_12:
        v = by_gt.get(cat)
        if v and v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:45s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
    print()

    all_pred_cats = sorted(set(
        pred for preds in confusion.values() for pred in preds
    ))
    # Short labels for readability
    short = {c: c[:12] for c in all_pred_cats}
    print("Confusion (row=GT, col=Pred, truncated labels):")
    header = f"{'GT':<20s}" + "".join(f"{short[c]:>13s}" for c in all_pred_cats)
    print(header)
    print("-" * len(header))
    for gt_cat in CATS_12:
        if gt_cat not in confusion:
            continue
        row = f"{gt_cat[:20]:<20s}"
        for pred_cat in all_pred_cats:
            row += f"{confusion[gt_cat].get(pred_cat, 0):>13d}"
        print(row)
    print("=" * 65)

    return {
        "overall_accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "unknown": unknown_count,
        "per_category": {
            cat: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0, **v}
            for cat, v in by_gt.items()
        },
        "confusion": dict(confusion),
        "details": details,
    }
