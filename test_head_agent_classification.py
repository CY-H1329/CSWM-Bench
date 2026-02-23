#!/usr/bin/env python3
"""
Head Agent classification accuracy test.

Measures how accurately the Head Agent (Qwen3-VL-4B) classifies
benchmark questions into the unified 16-category taxonomy.

Supports both CV-Bench and 3DSRBench.

Usage (CLI):
    python test_head_agent_classification.py \
        --benchmark cvbench \
        --max_samples 200 \
        --device cuda

Usage (Jupyter):
    from test_head_agent_classification import run_classification_test
    results = run_classification_test(benchmark="cvbench", max_samples=200)
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2.config import ALL_CATEGORIES, CATEGORY_DESCRIPTIONS
from src2.agents.mas_v2.prompts import build_head_agent_prompt
from src2.agents.mas_v2.pipeline import parse_category
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# GT category → unified taxonomy mapping
# ======================================================================
# CV-Bench task field uses capitalized names; 3DSRBench uses snake_case.
GT_TO_TAXONOMY = {
    # CV-Bench
    "Count": "count",
    "Relation": "relation",
    "Depth": "depth",
    "Distance": "distance",
    # 3DSRBench (already matches taxonomy)
    "location_above": "location_above",
    "height_higher": "height_higher",
    "location_closer_to_camera": "location_closer_to_camera",
    "multi_object_closer_to": "multi_object_closer_to",
    "orientation_on_the_left": "orientation_on_the_left",
    "multi_object_facing": "multi_object_facing",
    "multi_object_same_direction": "multi_object_same_direction",
    "orientation_in_front_of": "orientation_in_front_of",
    "multi_object_viewpoint_towards_object": "multi_object_viewpoint_towards_object",
    "orientation_viewpoint": "orientation_viewpoint",
    "location_next_to": "location_next_to",
    "multi_object_parallel": "multi_object_parallel",
}


def normalize_gt(raw_gt: str) -> Optional[str]:
    """Map benchmark GT category to unified taxonomy name."""
    if not raw_gt:
        return None
    raw_gt = raw_gt.strip()
    if raw_gt in GT_TO_TAXONOMY:
        return GT_TO_TAXONOMY[raw_gt]
    if raw_gt.lower() in {c.lower() for c in ALL_CATEGORIES}:
        for c in ALL_CATEGORIES:
            if c.lower() == raw_gt.lower():
                return c
    return None


# ======================================================================
# Build Head Agent runner
# ======================================================================
def build_head_runner(device: str = "cuda") -> Callable:
    """Load Qwen3-VL-4B and return head_generate(image, prompt)->str."""
    from src2.models.qwen3 import Qwen3Runner

    runner = Qwen3Runner(device=device)
    logger.info("Qwen3-VL-4B loaded on %s", device)

    def head_generate(image, prompt: str) -> str:
        return runner.generate(image, prompt, temperature=0.0, max_new_tokens=64)

    return head_generate


# ======================================================================
# Core test loop
# ======================================================================
def run_classification_test(
    benchmark: str = "cvbench",
    head_generate: Optional[Callable] = None,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run Head Agent classification test and return metrics.

    Args:
        benchmark: "cvbench" or "3dsrbench"
        head_generate: fn(image, prompt)->str. If None, loads Qwen3-VL-4B.
        device: torch device for model loading.
        max_samples: limit number of samples (None = all).
        seed: random seed for sampling.
        output_dir: if set, save detailed results.

    Returns:
        dict with overall accuracy, per-category accuracy, confusion, details.
    """
    if head_generate is None:
        head_generate = build_head_runner(device)

    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    total = len(dataset)
    logger.info("Loaded %d samples from %s", total, benchmark)

    details = []
    correct_count = 0
    by_gt = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for i in range(total):
        example = dataset[i]
        image = get_benchmark_image(example, benchmark)
        query = get_benchmark_prompt(example, benchmark)
        raw_gt = get_benchmark_category(example, benchmark)
        gt_cat = normalize_gt(raw_gt)

        if image is None:
            logger.warning("Sample %d: image is None, skipping.", i)
            continue

        if gt_cat is None:
            logger.warning("Sample %d: unknown GT category '%s', skipping.", i, raw_gt)
            continue

        prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
        head_raw = head_generate(image, prompt)
        pred_cat = parse_category(head_raw, ALL_CATEGORIES)

        is_correct = pred_cat == gt_cat
        if is_correct:
            correct_count += 1
        by_gt[gt_cat]["total"] += 1
        if is_correct:
            by_gt[gt_cat]["correct"] += 1
        confusion[gt_cat][pred_cat] += 1

        details.append({
            "index": i,
            "query": query[:200],
            "gt_category": gt_cat,
            "pred_category": pred_cat,
            "head_raw": head_raw.strip()[:100],
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or i == total - 1:
            evaluated = len(details)
            acc = correct_count / evaluated if evaluated > 0 else 0
            logger.info(
                "Progress %d/%d | evaluated: %d | running acc: %.1f%%",
                i + 1, total, evaluated, 100 * acc,
            )

    # --- Metrics ---
    evaluated = len(details)
    overall_acc = correct_count / evaluated if evaluated > 0 else 0
    per_category = {}
    for cat in sorted(by_gt.keys()):
        v = by_gt[cat]
        per_category[cat] = {
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
            "correct": v["correct"],
            "total": v["total"],
        }

    confusion_dict = {
        gt: dict(sorted(preds.items()))
        for gt, preds in sorted(confusion.items())
    }

    result = {
        "benchmark": benchmark,
        "total_samples": total,
        "evaluated": evaluated,
        "overall_accuracy": overall_acc,
        "correct": correct_count,
        "per_category": per_category,
        "confusion_matrix": confusion_dict,
    }

    # --- Print report ---
    print("\n" + "=" * 60)
    print(f"HEAD AGENT CLASSIFICATION TEST — {benchmark.upper()}")
    print("=" * 60)
    print(f"Samples evaluated: {evaluated}/{total}")
    print(f"Overall accuracy:  {100 * overall_acc:.1f}%  ({correct_count}/{evaluated})")
    print()
    print("Per-category accuracy:")
    for cat, v in per_category.items():
        print(f"  {cat:45s}  {100*v['accuracy']:5.1f}%  ({v['correct']}/{v['total']})")
    print()
    print("Confusion matrix (rows=GT, cols=predicted):")
    all_pred_cats = sorted(set(
        pred for preds in confusion.values() for pred in preds
    ))
    header = f"{'GT \\ Pred':<30s}" + "".join(f"{c[:12]:>13s}" for c in all_pred_cats)
    print(header)
    print("-" * len(header))
    for gt_cat in sorted(confusion.keys()):
        row = f"{gt_cat:<30s}"
        for pred in all_pred_cats:
            cnt = confusion[gt_cat].get(pred, 0)
            row += f"{cnt:>13d}"
        print(row)
    print("=" * 60)

    # --- Save ---
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / f"head_agent_test_{benchmark}_{ts}"
        out_path.mkdir(parents=True, exist_ok=True)

        (out_path / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        with open(out_path / "details.jsonl", "w") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        logger.info("Results saved to %s", out_path)
        result["output_path"] = str(out_path)

    result["details"] = details
    return result


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Head Agent classification accuracy test")
    parser.add_argument("--benchmark", choices=["cvbench", "3dsrbench"], default="cvbench")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/head_agent_test")
    args = parser.parse_args()

    run_classification_test(
        benchmark=args.benchmark,
        device=args.device,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
