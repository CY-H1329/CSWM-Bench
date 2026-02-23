#!/usr/bin/env python3
"""
Head Agent classification test — 3DSRBench with GROUPED categories.

Shows both fine-grained (12-cat) and grouped accuracy.
If the model picks a category within the same group as GT → counted as correct.

Usage (Jupyter):
    from test_head_grouped_3dsrbench import run_grouped_test
    results = run_grouped_test(runner, max_samples=200)
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
    "location_above", "height_higher",
    "location_closer_to_camera", "multi_object_closer_to",
    "location_next_to",
    "orientation_on_the_left", "orientation_in_front_of", "orientation_viewpoint",
    "multi_object_facing", "multi_object_same_direction",
    "multi_object_viewpoint_towards_object",
    "multi_object_parallel",
]

# --- Grouping: semantically similar categories merged ---
GROUPS = {
    "vertical":     ["location_above", "height_higher"],
    "camera_dist":  ["location_closer_to_camera", "multi_object_closer_to"],
    "adjacency":    ["location_next_to"],
    "orientation":  [
        "orientation_on_the_left", "orientation_in_front_of",
        "orientation_viewpoint", "multi_object_facing",
        "multi_object_same_direction", "multi_object_viewpoint_towards_object",
    ],
    "alignment":    ["multi_object_parallel"],
}

CAT_TO_GROUP = {}
for group, cats in GROUPS.items():
    for c in cats:
        CAT_TO_GROUP[c] = group


def parse_category_12(raw):
    raw_clean = (raw or "").strip().lower()
    for cat in CATS_12:
        if cat == raw_clean:
            return cat
    for cat in CATS_12:
        if cat in raw_clean:
            return cat
    return "UNKNOWN"


def run_grouped_test(runner, max_samples=None, seed=42):
    dataset = load_benchmark("3dsrbench", max_samples=max_samples, seed=seed)

    # Fine-grained stats
    fine_correct = 0
    fine_total = 0
    fine_by_gt = defaultdict(lambda: {"correct": 0, "total": 0})

    # Grouped stats
    group_correct = 0
    group_by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    group_confusion = defaultdict(lambda: defaultdict(int))

    unknown_count = 0
    details = []

    for i in range(len(dataset)):
        ex = dataset[i]
        image = get_benchmark_image(ex, "3dsrbench")
        query = get_benchmark_prompt(ex, "3dsrbench")
        raw_gt = (get_benchmark_category(ex, "3dsrbench") or "").strip()
        if image is None or raw_gt not in CATS_12:
            continue

        prompt = build_head_agent_prompt(query, CATS_12, CATEGORY_DESCRIPTIONS)
        raw = runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)
        pred = parse_category_12(raw)

        if pred == "UNKNOWN":
            unknown_count += 1

        fine_total += 1
        gt_group = CAT_TO_GROUP.get(raw_gt, "UNKNOWN")
        pred_group = CAT_TO_GROUP.get(pred, "UNKNOWN")

        # Fine-grained accuracy
        fine_hit = pred == raw_gt
        if fine_hit:
            fine_correct += 1
        fine_by_gt[raw_gt]["total"] += 1
        if fine_hit:
            fine_by_gt[raw_gt]["correct"] += 1

        # Grouped accuracy
        group_hit = gt_group == pred_group
        if group_hit:
            group_correct += 1
        group_by_gt[gt_group]["total"] += 1
        if group_hit:
            group_by_gt[gt_group]["correct"] += 1
        group_confusion[gt_group][pred_group] += 1

        details.append({
            "index": i, "gt": raw_gt, "pred": pred,
            "gt_group": gt_group, "pred_group": pred_group,
            "fine_correct": fine_hit, "group_correct": group_hit,
            "raw": raw.strip()[:80],
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress {i+1}/{len(dataset)} | fine: {100*fine_correct/fine_total:.1f}% | grouped: {100*group_correct/fine_total:.1f}%")

    if fine_total == 0:
        print("No samples evaluated!")
        return {}

    # --- Report ---
    print()
    print("=" * 70)
    print("3DSRBench — FINE-GRAINED (12 categories)")
    print("=" * 70)
    print(f"Overall: {fine_correct}/{fine_total} = {100*fine_correct/fine_total:.1f}%")
    if unknown_count:
        print(f"UNKNOWN: {unknown_count}")
    print()
    for cat in CATS_12:
        v = fine_by_gt.get(cat)
        if v and v["total"] > 0:
            acc = v["correct"] / v["total"]
            grp = CAT_TO_GROUP[cat]
            print(f"  [{grp:12s}] {cat:45s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")

    print()
    print("=" * 70)
    print("3DSRBench — GROUPED (5 groups)")
    print("=" * 70)
    print(f"Overall: {group_correct}/{fine_total} = {100*group_correct/fine_total:.1f}%")
    print()
    group_order = ["vertical", "camera_dist", "adjacency", "orientation", "alignment"]
    for grp in group_order:
        v = group_by_gt.get(grp)
        if v and v["total"] > 0:
            acc = v["correct"] / v["total"]
            members = ", ".join(GROUPS[grp])
            print(f"  {grp:15s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
            print(f"    members: {members}")
            print()

    print("Group confusion (row=GT, col=Pred):")
    all_groups = group_order + (["UNKNOWN"] if unknown_count else [])
    header = f"{'':15s}" + "".join(f"{g:>15s}" for g in all_groups)
    print(header)
    print("-" * len(header))
    for gt_grp in group_order:
        row = f"{gt_grp:15s}"
        for pred_grp in all_groups:
            row += f"{group_confusion[gt_grp].get(pred_grp, 0):>15d}"
        print(row)
    print("=" * 70)

    return {
        "fine_accuracy": fine_correct / fine_total,
        "grouped_accuracy": group_correct / fine_total,
        "fine_correct": fine_correct,
        "group_correct": group_correct,
        "total": fine_total,
        "unknown": unknown_count,
        "fine_per_category": dict(fine_by_gt),
        "group_per_group": dict(group_by_gt),
        "group_confusion": dict(group_confusion),
        "details": details,
    }
