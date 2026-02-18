#!/usr/bin/env python3
"""
Head-Agent Category Routing evaluation on CV-Bench.

Tests Head-Agent candidates (GPT-4o, Claude, Gemini) on routing questions
to the closest benchmark category. No answer prediction — only category selection.

Usage:
  python scripts/evals/head_agent_cvbench/run_eval_category_routing.py --max_samples 30
  python scripts/evals/head_agent_cvbench/run_eval_category_routing.py --max_samples 100 --model gpt5_2
  python scripts/evals/head_agent_cvbench/run_eval_category_routing.py --full_dataset

Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_category

# Prompt
from prompt import build_category_routing_prompt, get_categories

# API runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
ClaudeRunner = _runners.ClaudeRunner
GPT4oRunner = _runners.GPT4oRunner
OpenRouterRunner = _runners.OpenRouterRunner

def extract_predicted_category(response: str, categories: list) -> str:
    """Extract category from model output. Normalize to match GT."""
    if not response or not response.strip():
        return ""
    text = response.strip()
    # Match "Category: X" or "category: x"
    m = re.search(r"Category\s*:\s*([A-Za-z][A-Za-z0-9_\s\-]*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        for c in categories:
            if c.lower().replace(" ", "_").replace("-", "_") == raw or raw in c.lower():
                return c
        return m.group(1).strip()
    # Fallback: first line that matches a valid category
    for line in text.split("\n"):
        line_clean = line.strip().lower().replace(" ", "_").replace("-", "_")
        for c in categories:
            if c.lower().replace(" ", "_") in line_clean or line_clean in c.lower():
                return c
    return ""


def normalize_gt_category(gt: str, categories: list) -> str:
    """Normalize GT category to match benchmark categories."""
    if not gt:
        return ""
    s = str(gt).strip()
    for c in categories:
        if c.lower() == s.lower():
            return c
    return s


def get_runner(model_key: str, config: dict):
    cfg = config.get("models", {}).get(model_key, {})
    if not cfg.get("enabled", True):
        return None
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key:
        return None
    model_id = cfg.get("model_id", "")
    if model_key == "claude_opus_4_5":
        return ClaudeRunner(model_id=model_id, api_key=api_key)
    if model_key == "gpt5_2":
        return GPT4oRunner(model_id=model_id, api_key=api_key)
    if model_key == "glm5":
        base_url = cfg.get("base_url", "https://openrouter.ai/api/v1")
        return OpenRouterRunner(model_id=model_id, api_key=api_key, base_url=base_url)
    return None


def main():
    parser = argparse.ArgumentParser(description="Head-Agent Category Routing on CV-Bench")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--full_dataset", action="store_true")
    parser.add_argument("--model", choices=["gpt5_2", "glm5", "claude_opus_4_5"])
    parser.add_argument("--benchmark", choices=["cvbench", "3dsrbench"], default="cvbench")
    parser.add_argument("--text_only", action="store_true", help="Question only (no image) for routing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {})
    max_samples = None if args.full_dataset else (args.max_samples or ds_cfg.get("max_samples", 100))
    seed = args.seed or ds_cfg.get("seed", 42)
    output_dir = Path(config.get("output", {}).get("dir", "results"))

    benchmark = args.benchmark
    subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / "head_agent" / benchmark / "category_routing" / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = run_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {benchmark}... (max_samples={max_samples or 'all'}, seed={seed})")
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    print(f"  {len(dataset)} samples")

    model_keys = [args.model] if args.model else ["gpt5_2", "glm5", "claude_opus_4_5"]
    results_table = []

    for model_key in model_keys:
        runner = get_runner(model_key, config)
        if runner is None:
            print(f"\n[skip] {model_key} (disabled or missing API key)")
            continue

        print(f"\n--- {model_key} ---")
        preds = []
        gt_list = []
        details = []

        categories = get_categories(benchmark)
        for i in tqdm(range(len(dataset)), desc=model_key):
            example = dataset[i]
            query = get_benchmark_prompt(example, benchmark)
            gt = get_benchmark_category(example, benchmark)
            gt_norm = normalize_gt_category(gt, categories) if gt else ""

            prompt = build_category_routing_prompt(query, benchmark)

            try:
                from src.benchmarks import get_benchmark_image
                image = get_benchmark_image(example, benchmark)
                if args.text_only or image is None:
                    from PIL import Image
                    image = Image.new("RGB", (1, 1), color="white")
                response = runner.generate(image, prompt, temperature=0.0, max_tokens=args.max_tokens)

                pred = extract_predicted_category(response, categories)
                preds.append(pred)
                gt_list.append(gt_norm)
                details.append({
                    "idx": i,
                    "query": query[:200],
                    "gt": gt_norm,
                    "pred": pred,
                    "correct": pred and gt_norm and pred.lower() == gt_norm.lower(),
                    "full_response": response,
                })
                with open(responses_dir / f"{model_key}_sample_{i:05d}.txt", "w", encoding="utf-8") as f:
                    f.write(f"=== QUERY ===\n{query[:500]}...\n\n=== GT ===\n{gt_norm}\n\n=== PRED ===\n{pred}\n\n=== FULL ===\n{response}\n")
            except Exception as e:
                preds.append("")
                gt_list.append(gt_norm)
                details.append({"idx": i, "error": str(e), "gt": gt_norm, "pred": ""})

        # Filter to samples with valid GT for accuracy
        valid_pairs = [(p, g) for p, g in zip(preds, gt_list) if g]
        acc = sum(1 for p, g in valid_pairs if p and p.lower() == g.lower()) / len(valid_pairs) if valid_pairs else 0.0

        # Per-category accuracy
        by_cat = {}
        for d in details:
            g = d.get("gt", "")
            if not g:
                continue
            if g not in by_cat:
                by_cat[g] = {"correct": 0, "total": 0}
            by_cat[g]["total"] += 1
            if d.get("correct"):
                by_cat[g]["correct"] += 1

        model_dir = run_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(model_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": model_key,
                "metric": "category_routing_accuracy",
                "accuracy": acc,
                "n": len(valid_pairs),
                "by_category": {k: v["correct"] / v["total"] if v["total"] else 0 for k, v in by_cat.items()},
                "by_category_n": {k: v["total"] for k, v in by_cat.items()},
            }, f, indent=2)

        results_table.append({"model": model_key, "accuracy": acc, "n": len(valid_pairs)})
        print(f"  Category Routing Accuracy: {acc:.4f} ({len(valid_pairs)} samples)")
        for cat, v in sorted(by_cat.items()):
            c_acc = v["correct"] / v["total"] if v["total"] else 0
            print(f"    {cat}: {c_acc:.2f} ({v['correct']}/{v['total']})")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Head-Agent Category Routing — {benchmark.upper()}\n\n")
        f.write("| Model | Accuracy | N |\n|-------|----------|---|\n")
        for r in results_table:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['n']} |\n")

    print("\n" + "=" * 60)
    print("Head-Agent Category Routing — Summary")
    print("=" * 60)
    for r in results_table:
        print(f"  {r['model']}: {r['accuracy']:.4f}")
    print("=" * 60)
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
