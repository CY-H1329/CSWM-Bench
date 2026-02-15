#!/usr/bin/env python3
"""
Évaluation 3DSRBench — Qwen3-4B uniquement.
Exécution séparée pour éviter toute interférence entre modèles.
Utilise les catégories réelles 3DSRBench: Height, Location, Orientation, Multi-Object.

Usage:
  python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py
  python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 50 --seed 42
"""
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import yaml
from tqdm import tqdm

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
)
from src.data import normalize_answer_only, accuracy, extract_predicted_category, normalize_category
from src.models.qwen3 import Qwen3Runner

# Import common (works when run directly or as module)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).parent / "common.py"
)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
build_spatial_prompt = _common.build_spatial_prompt


def load_config(path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / path
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="3DSRBench eval — Qwen3-4B only")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    output_dir = Path(config.get("output", {}).get("dir", "results"))
    benchmark = "3dsrbench"
    model_name = "qwen3_4b"

    if eval_cfg.get("use_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / benchmark / "qwen3_4b" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = run_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {benchmark}... (seed={args.seed})")
    dataset = load_benchmark(benchmark, max_samples=args.max_samples, seed=args.seed)
    print(f"  {len(dataset)} samples")

    m_cfg = config.get("models", {}).get("qwen3_4b", {})
    if not m_cfg.get("enabled", True):
        raise RuntimeError("qwen3_4b is disabled in config")
    runner = Qwen3Runner(
        model_id=m_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        device=m_cfg.get("device", "cuda"),
        use_flash_attn=eval_cfg.get("use_flash_attn", True),
    )
    print(f"  [load] {model_name} ({runner.model_id})")

    preds = []
    gt_list = []
    details = []

    for i in tqdm(range(len(dataset)), desc=model_name):
        example = dataset[i]
        image = get_benchmark_image(example, benchmark)
        query = get_benchmark_prompt(example, benchmark)
        gt = get_benchmark_answer(example, benchmark)
        category = get_benchmark_category(example, benchmark) or "unknown"

        if image is None:
            preds.append("")
            gt_list.append(gt)
            gt_cat = normalize_category(category) if category and category != "unknown" else ""
            details.append({"idx": i, "error": "no_image", "gt": gt, "category_gt": gt_cat})
            continue

        full_prompt = build_spatial_prompt(query)

        try:
            response = runner.generate(
                image,
                full_prompt,
                temperature=eval_cfg.get("mas_temperature", 0.0),
                max_new_tokens=args.max_new_tokens,
            )
            letter = normalize_answer_only(response)
            pred_category = extract_predicted_category(response)
            gt_category_norm = normalize_category(category) if category and category != "unknown" else ""
            preds.append(letter)
            gt_list.append(gt)
            details.append({
                "idx": i,
                "query": query,
                "gt": gt,
                "pred": letter,
                "category": category,
                "category_gt": gt_category_norm,
                "pred_category": pred_category,
                "full_response": response,
            })
            sample_path = responses_dir / f"sample_{i:05d}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("=== QUERY ===\n")
                f.write(query + "\n\n")
                f.write("=== GT ===\n")
                f.write(gt + "\n\n")
                f.write("=== CATEGORY GT / PRED ===\n")
                f.write(f"{gt_category_norm} / {pred_category}\n\n")
                f.write("=== FULL RESPONSE ===\n")
                f.write(response + "\n\n")
                f.write("=== EXTRACTED PRED ===\n")
                f.write(letter + "\n")
        except Exception as e:
            preds.append("")
            gt_list.append(gt)
            gt_cat = normalize_category(category) if category and category != "unknown" else ""
            details.append({"idx": i, "error": str(e), "gt": gt, "category_gt": gt_cat})

    acc = accuracy(preds, gt_list)

    # Category classification accuracy (GT vs predicted)
    cat_pairs = [(d.get("category_gt", ""), d.get("pred_category", "")) for d in details if d.get("category_gt")]
    cat_gt_list = [p[0] for p in cat_pairs]
    cat_pred_list = [p[1] for p in cat_pairs]
    cat_cls_acc = accuracy(cat_pred_list, cat_gt_list) if cat_pairs else 0.0
    cat_n = len(cat_pairs)

    with open(run_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    pred_dist = {k: v for k, v in sorted(Counter(p for p in preds if p).items())}
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "accuracy": acc,
            "n": len(details),
            "pred_distribution": pred_dist,
            "category_cls_accuracy": cat_cls_acc,
            "category_cls_n": cat_n,
        }, f, indent=2)

    pred_dist_str = ", ".join(f"{k}={v}" for k, v in sorted(pred_dist.items()))

    print("\n" + "=" * 50)
    print("3DSRBench — Qwen3-4B")
    print("=" * 50)
    print(f"Answer Accuracy: {acc:.4f} ({len(details)} samples)")
    print(f"Category Cls Accuracy: {cat_cls_acc:.4f} ({cat_n} samples with GT category)")
    print(f"Pred distribution: {pred_dist_str or 'N/A'}")
    print("=" * 50)
    print(f"Résultats: {run_dir}")

    for attr in ("model", "processor"):
        if hasattr(runner, attr):
            delattr(runner, attr)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
