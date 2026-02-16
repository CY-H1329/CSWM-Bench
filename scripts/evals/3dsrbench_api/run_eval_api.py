#!/usr/bin/env python3
"""
3DSRBench — API models (Claude Sonnet 4.5, GPT-4o, DeepSeek-VL, Gemini Robotics-ER).
100 samples, separate from GPU eval. Does not modify existing code.

Usage:
  python scripts/evals/3dsrbench_api/run_eval_api.py
  python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 50
  python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset   # dataset complet → full_dataset/

Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY
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
from src.data import normalize_answer_only, accuracy, extract_predicted_category, normalize_category

# Import prompt from common (sibling)
import importlib.util
_common_path = ROOT / "scripts/evals/3dsrbench/common.py"
_spec = importlib.util.spec_from_file_location("common", _common_path)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
build_spatial_prompt = _common.build_spatial_prompt

# Import API runners (same directory)
_runners_path = Path(__file__).parent / "runners.py"
_runners_spec = importlib.util.spec_from_file_location("runners", _runners_path)
_runners = importlib.util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
ClaudeRunner = _runners.ClaudeRunner
GPT4oRunner = _runners.GPT4oRunner
DeepSeekVLRunner = _runners.DeepSeekVLRunner
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
    if model_key == "deepseek_vl":
        return DeepSeekVLRunner(
            model_id=model_id,
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
        )
    if model_key == "gemini_robotics_er":
        return GeminiRunner(model_id=model_id, api_key=api_key)
    return None


def main():
    parser = argparse.ArgumentParser(description="3DSRBench API models")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_api.yaml"))
    parser.add_argument("--max_samples", type=int, default=None, help="Limiter à N samples (défaut: 100 si pas --full_dataset)")
    parser.add_argument("--full_dataset", action="store_true", help="Dataset complet, sortie dans full_dataset/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {})
    use_full = args.full_dataset
    max_samples = None if use_full else (args.max_samples or ds_cfg.get("max_samples", 1000))
    seed = args.seed or ds_cfg.get("seed", 42)
    output_dir = Path(config.get("output", {}).get("dir", "results"))

    subdir = "full_dataset" if use_full else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / "3dsrbench" / "api_models" / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 3DSRBench... (max_samples={'all' if use_full else max_samples}, seed={seed})")
    dataset = load_benchmark("3dsrbench", max_samples=max_samples, seed=seed)
    print(f"  {len(dataset)} samples")

    model_keys = ["claude_sonnet_4_5", "gpt4o", "deepseek_vl", "gemini_robotics_er"]
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
                image = get_benchmark_image(example, "3dsrbench")
                query = get_benchmark_prompt(example, "3dsrbench")
                gt = get_benchmark_answer(example, "3dsrbench")
                category = get_benchmark_category(example, "3dsrbench") or "unknown"
                gt_category_norm = normalize_category(category) if category and category != "unknown" else ""

                if image is None:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({"idx": i, "error": "no_image", "gt": gt, "category_gt": gt_category_norm})
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
                    pred_category = extract_predicted_category(response)
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
                    with open(responses_dir / f"sample_{i:05d}.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== QUERY ===\n{query}\n\n=== GT ===\n{gt}\n\n")
                        f.write(f"=== CATEGORY GT / PRED ===\n{gt_category_norm} / {pred_category}\n\n")
                        f.write(f"=== FULL RESPONSE ===\n{response}\n\n=== EXTRACTED PRED ===\n{letter}\n")
                except Exception as e:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({"idx": i, "error": str(e), "gt": gt, "category_gt": gt_category_norm})

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

    # Summary
    n_total = len(dataset)
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# 3DSRBench — API Models ({n_total} samples)\n")
        f.write("# Chaque modèle : with_prompt (spatial) et without_prompt (question seule)\n\n")
        f.write("| Model | Answer Acc | Category Cls Acc | N |\n")
        f.write("|-------|------------|------------------|---|\n")
        for r in results_table:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['category_cls_acc']:.4f} | {r['n']} |\n")

    print("\n" + "=" * 60)
    print("3DSRBench — API Models Summary")
    print("=" * 60)
    for r in results_table:
        print(f"  {r['model']}: Answer={r['accuracy']:.4f}, CatCls={r['category_cls_acc']:.4f}")
    print("=" * 60)
    print(f"Résultats: {run_dir}")


if __name__ == "__main__":
    main()
