#!/usr/bin/env python3
"""
Multi-Agent System evaluation: Head → Perception → Reasoning.
- Head-Agent: classifie le task (ne reçoit PAS la catégorie)
- Perception Agent: reçoit query + task_class
- Reasoning Agent: reçoit query + task_class + perception → answer

Usage:
  python run_eval_mas.py --benchmark stvqa7k --head qwen --perception qwen --reasoning qwen
  python run_eval_mas.py --benchmark stvqa7k --head llava --perception llava --reasoning llava
  python run_eval_mas.py --benchmark stvqa7k --head qwen --perception llava --reasoning qwen
  python run_eval_mas.py --benchmark stvqa7k --max_per_category 8  # quick test
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm

from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_answer, get_benchmark_image, get_benchmark_category
from src.agents import run_mas_pipeline
from src.data import normalize_answer_only, accuracy
from src.models.qwen import QwenRunner
from src.models.qwen3 import Qwen3Runner
from src.models.llava import LLaVARunner
from src.models.sa2va import Sa2VARunner
from src.models.gpt import GPTRunner
from src.models.gemini import GeminiRunner


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_runner(model_name: str, config: dict):
    """Get model runner. Supports: qwen, qwen3_4b, llava, llava4d, sa2va, gpt, gemini."""
    models_cfg = config.get("models", {})

    if model_name == "qwen":
        m_cfg = models_cfg.get("qwen", {})
        if not m_cfg.get("enabled", True):
            return None
        return QwenRunner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "qwen3_4b":
        m_cfg = models_cfg.get("qwen3_4b", {})
        if not m_cfg.get("enabled", True):
            return None
        return Qwen3Runner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "llava":
        m_cfg = models_cfg.get("llava", {})
        if not m_cfg.get("enabled", True):
            return None
        return LLaVARunner(
            model_id=m_cfg.get("model_id", "llava-hf/llava-v1.6-mistral-7b-hf"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "llava4d":
        m_cfg = models_cfg.get("llava4d", {})
        if not m_cfg.get("enabled", True):
            return None
        return LLaVARunner(
            model_id=m_cfg.get("model_id", "llava-hf/llava-v1.6-mistral-7b-hf"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "sa2va":
        m_cfg = models_cfg.get("sa2va", {})
        if not m_cfg.get("enabled", True):
            return None
        return Sa2VARunner(
            model_id=m_cfg.get("model_id", "ByteDance/Sa2VA-4B"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "gpt":
        m_cfg = config.get("models", {}).get("gpt", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            print(f"[skip] {model_name}: no API key")
            return None
        return GPTRunner(model_id=m_cfg.get("model_id", "gpt-4o"), api_key=api_key)
    elif model_name == "gemini":
        m_cfg = config.get("models", {}).get("gemini", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(f"[skip] {model_name}: no API key")
            return None
        return GeminiRunner(model_id=m_cfg.get("model_id", "gemini-2.0-flash"), api_key=api_key)
    raise ValueError(f"Unknown model: {model_name}")


def make_generate_fn(runner, eval_cfg: dict):
    """Wrap runner.generate with config."""
    temp = eval_cfg.get("temperature", 0.0)
    max_new = eval_cfg.get("max_new_tokens", 512)
    top_k = eval_cfg.get("top_k", 0)
    top_p = eval_cfg.get("top_p", 0.0)

    def _generate(image, prompt):
        return runner.generate(
            image, prompt,
            temperature=temp,
            max_new_tokens=max_new,
            top_k=top_k,
            top_p=top_p,
        )
    return _generate


def main():
    parser = argparse.ArgumentParser(description="MAS evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--benchmark", default="stvqa7k", choices=["stvqa7k", "omni3d", "cvbench", "3dsrbench"])
    parser.add_argument(
        "--head", default="qwen3_4b",
        help="Head-Agent model: qwen, qwen3_4b, llava, llava4d, sa2va",
    )
    parser.add_argument(
        "--perception", default="qwen3_4b",
        help="Perception-Agent model: qwen, qwen3_4b, llava, llava4d, sa2va",
    )
    parser.add_argument(
        "--reasoning", default="qwen3_4b",
        help="Reasoning-Agent model: qwen, qwen3_4b, llava, llava4d, sa2va",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_category", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    output_dir = Path(config.get("output", {}).get("dir", "results"))

    # Output: runs/<benchmark>/<head>_<perception>_<reasoning>/
    run_name = f"{args.head}_{args.perception}_{args.reasoning}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / args.benchmark / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {args.benchmark}...")
    dataset = load_benchmark(
        args.benchmark,
        max_samples=args.max_samples,
        max_per_category=args.max_per_category,
    )
    print(f"  {len(dataset)} samples")

    # Create runners for each role
    head_runner = get_runner(args.head, config)
    perc_runner = get_runner(args.perception, config)
    reas_runner = get_runner(args.reasoning, config)
    if not all([head_runner, perc_runner, reas_runner]):
        raise RuntimeError("All three agents must be enabled")

    head_gen = make_generate_fn(head_runner, eval_cfg)
    perc_gen = make_generate_fn(perc_runner, eval_cfg)
    reas_gen = make_generate_fn(reas_runner, eval_cfg)

    # Run pipeline
    preds = []
    gt_list = []
    details = []

    for i in tqdm(range(len(dataset)), desc="MAS"):
        example = dataset[i]
        image = get_benchmark_image(example, args.benchmark)
        query = get_benchmark_prompt(example, args.benchmark)
        gt = get_benchmark_answer(example, args.benchmark)
        category = get_benchmark_category(example, args.benchmark)

        if image is None:
            preds.append("")
            gt_list.append(gt)
            details.append({"idx": i, "error": "no_image"})
            continue

        try:
            out = run_mas_pipeline(
                image, query,
                generate_fn=head_gen,
                head_model_fn=head_gen,
                perception_model_fn=perc_gen,
                reasoning_model_fn=reas_gen,
            )
            final = out["final_answer"]
            # Letter-based benchmarks: extract (A)/(B)/...
            if args.benchmark in ("stvqa7k", "cvbench", "3dsrbench") and len(gt) == 1 and gt in "ABCDEF":
                letter = normalize_answer_only(final)
            else:
                # Free-form: take last line or strip "Answer:"
                letter = final.strip()
                if "answer:" in letter.lower():
                    letter = letter.split("answer:")[-1].strip()
                if "\n" in letter:
                    letter = letter.split("\n")[-1].strip()
            preds.append(letter)
            gt_list.append(gt)
            details.append({
                "idx": i,
                "task_class": out["task_class"],
                "perception": out["perception_output"][:200],
                "pred": letter,
                "gt": gt,
                "category": category,
            })
        except Exception as e:
            preds.append("")
            gt_list.append(gt)
            details.append({"idx": i, "error": str(e)})

    # Compute accuracy
    if args.benchmark in ("stvqa7k", "cvbench", "3dsrbench"):
        acc = accuracy(preds, gt_list)
    else:
        # Omni3D: normalize for comparison (lowercase, strip)
        def _norm(s):
            return str(s).strip().lower()
        acc = sum(_norm(p) == _norm(g) for p, g in zip(preds, gt_list)) / len(gt_list) if gt_list else 0.0

    results = {
        "benchmark": args.benchmark,
        "combination": run_name,
        "accuracy": acc,
        "num_samples": len(gt_list),
        "head": args.head,
        "perception": args.perception,
        "reasoning": args.reasoning,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(run_dir / "details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\nAccuracy: {acc:.4f} ({len(gt_list)} samples)")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
