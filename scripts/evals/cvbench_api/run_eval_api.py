#!/usr/bin/env python3
"""
CV-Bench — API models (Claude Sonnet 4.5, GPT-4o, Gemini Robotics-ER).

1. Test (무료 API): --test  → 10 samples, all 3 models, with/without prompt
2. Full dataset: --full_dataset → all ~2638 samples

Usage:
  python scripts/evals/cvbench_api/run_eval_api.py --test
  python scripts/evals/cvbench_api/run_eval_api.py --full_dataset
  python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
  python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt

Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
)
from src.data import normalize_answer_only, accuracy, extract_predicted_category

# CV-Bench prompt
import importlib.util
_common_path = ROOT / "scripts/evals/cvbench/common.py"
_spec = importlib.util.spec_from_file_location("common", _common_path)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
build_spatial_prompt = _common.build_spatial_prompt
CVBENCH_TASK_CATEGORIES = _common.CVBENCH_TASK_CATEGORIES
CVBENCH_CATS = frozenset(c.lower().replace(" ", "_") for c in CVBENCH_TASK_CATEGORIES)

# API runners (reuse from 3dsrbench_api)
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = importlib.util.spec_from_file_location("runners", _runners_path)
_runners = importlib.util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
ClaudeRunner = _runners.ClaudeRunner
GPT4oRunner = _runners.GPT4oRunner
GeminiRunner = _runners.GeminiRunner


def get_runner(model_key: str, config: dict):
    cfg = config.get("models", {}).get(model_key, {})
    if not cfg.get("enabled", True):
        return None
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key:
        return None
    model_id = cfg.get("model_id", "")
    if model_key == "claude_sonnet_4_5":
        return ClaudeRunner(model_id=model_id, api_key=api_key)
    if model_key == "gpt4o":
        return GPT4oRunner(model_id=model_id, api_key=api_key)
    if model_key == "gemini_robotics_er":
        return GeminiRunner(model_id=model_id, api_key=api_key)
    return None


def main():
    parser = argparse.ArgumentParser(description="CV-Bench API models")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_api.yaml"))
    parser.add_argument("--test", action="store_true", help="무료 API 테스트: 10 samples, all models")
    parser.add_argument("--full_dataset", action="store_true", help="Full dataset (~2638 samples)")
    parser.add_argument("--model", choices=["claude_sonnet_4_5", "gpt4o", "gemini_robotics_er"], help="Single model")
    parser.add_argument("--without_prompt", action="store_true")
    parser.add_argument("--prompt_variant", choices=["with_prompt", "without_prompt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {})
    use_full = args.full_dataset
    max_samples = ds_cfg.get("test_samples", 10) if args.test else (None if use_full else ds_cfg.get("max_samples", 100))
    seed = args.seed or ds_cfg.get("seed", 42)
    output_dir = Path(config.get("output", {}).get("dir", "results"))

    subdir = "test" if args.test else ("full_dataset" if use_full else datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = output_dir / "runs" / "cvbench" / "api_models" / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CV-Bench... (max_samples={'all' if use_full else max_samples}, seed={seed})")
    dataset = load_benchmark("cvbench", max_samples=max_samples, seed=seed)
    print(f"  {len(dataset)} samples")

    model_keys = [args.model] if args.model else ["claude_sonnet_4_5", "gpt4o", "gemini_robotics_er"]
    if args.prompt_variant == "with_prompt":
        prompt_variants = [("with_prompt", lambda q: build_spatial_prompt(q))]
    elif args.without_prompt or args.prompt_variant == "without_prompt":
        prompt_variants = [("without_prompt", lambda q: q)]
    else:
        prompt_variants = [
            ("with_prompt", lambda q: build_spatial_prompt(q)),
            ("without_prompt", lambda q: q),
        ]

    results_table = []

    for model_key in model_keys:
        runner = get_runner(model_key, config)
        if runner is None:
            print(f"\n[skip] {model_key} (disabled or missing API key)")
            continue

        for variant_name, prompt_fn in prompt_variants:
            run_key = f"{model_key}_{variant_name}"
            print(f"\n--- {run_key} ---")
            model_dir = run_dir / run_key
            model_dir.mkdir(parents=True, exist_ok=True)
            responses_dir = model_dir / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)

            preds = []
            gt_list = []
            details = []

            for i in tqdm(range(len(dataset)), desc=run_key):
                example = dataset[i]
                image = get_benchmark_image(example, "cvbench")
                query = get_benchmark_prompt(example, "cvbench")
                gt = get_benchmark_answer(example, "cvbench")
                category = get_benchmark_category(example, "cvbench") or "unknown"
                gt_cat = str(category).strip().lower().replace(" ", "_") if category and category != "unknown" else ""

                if image is None:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({"idx": i, "error": "no_image", "gt": gt, "category_gt": gt_cat})
                    continue

                full_prompt = prompt_fn(query)

                try:
                    response = runner.generate(
                        image,
                        full_prompt,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    )
                    letter = normalize_answer_only(response)
                    pred_category = extract_predicted_category(response, CVBENCH_CATS)
                    preds.append(letter)
                    gt_list.append(gt)
                    details.append({
                        "idx": i,
                        "query": query,
                        "gt": gt,
                        "pred": letter,
                        "category": category,
                        "category_gt": gt_cat,
                        "pred_category": pred_category,
                        "full_response": response,
                    })
                    with open(responses_dir / f"sample_{i:05d}.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== QUERY ===\n{query}\n\n=== GT ===\n{gt}\n\n")
                        f.write(f"=== CATEGORY GT / PRED ===\n{gt_cat} / {pred_category}\n\n")
                        f.write(f"=== FULL RESPONSE ===\n{response}\n\n=== EXTRACTED PRED ===\n{letter}\n")
                except Exception as e:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({"idx": i, "error": str(e), "gt": gt, "category_gt": gt_cat})

            acc = accuracy(preds, gt_list)
            cat_pairs = [(d.get("category_gt", ""), d.get("pred_category", "")) for d in details if d.get("category_gt")]
            cat_cls_acc = accuracy([p[1] for p in cat_pairs], [p[0] for p in cat_pairs]) if cat_pairs else 0.0
            pred_dist = {k: v for k, v in sorted(Counter(p for p in preds if p).items())}

            with open(model_dir / "details.jsonl", "w", encoding="utf-8") as f:
                for d in details:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            with open(model_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump({
                    "model": run_key,
                    "prompt_variant": variant_name,
                    "accuracy": acc,
                    "n": len(details),
                    "pred_distribution": pred_dist,
                    "category_cls_accuracy": cat_cls_acc,
                    "category_cls_n": len(cat_pairs),
                }, f, indent=2)

            results_table.append({
                "model": run_key,
                "accuracy": acc,
                "category_cls_acc": cat_cls_acc,
                "n": len(details),
            })
            print(f"  Answer Accuracy: {acc:.4f} | Category Cls: {cat_cls_acc:.4f} | N={len(details)}")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# CV-Bench — API Models ({len(dataset)} samples)\n\n")
        f.write("| Model | Answer Acc | Category Cls Acc | N |\n")
        f.write("|-------|------------|------------------|---|\n")
        for r in results_table:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['category_cls_acc']:.4f} | {r['n']} |\n")

    print("\n" + "=" * 60)
    print("CV-Bench — API Models Summary")
    print("=" * 60)
    for r in results_table:
        print(f"  {r['model']}: Answer={r['accuracy']:.4f}, CatCls={r['category_cls_acc']:.4f}")
    print("=" * 60)
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
