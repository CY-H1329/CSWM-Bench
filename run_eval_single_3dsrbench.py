#!/usr/bin/env python3
"""
Évaluation single-agent sur 3DSRBench: Qwen3-4B, Sa2VA, LLaVA4D.
Chaque modèle reçoit image + query avec le prompt de raisonnement spatial.
Sauvegarde: réponse complète, GT, démarche. Tableau final par accuracy.

Usage:
  python run_eval_single_3dsrbench.py
  python run_eval_single_3dsrbench.py --max_samples 50 --seed 42
"""
import argparse
import gc
import json
import os
from pathlib import Path
from datetime import datetime

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
from src.data import normalize_answer_only, accuracy
from src.models.qwen3 import Qwen3Runner
from src.models.llava import LLaVARunner
from src.models.sa2va import Sa2VARunner


SPATIAL_REASONING_PROMPT = """# ROLE
You are an expert in spatial reasoning.
Your objective is to solve visual spatial reasoning tasks accurately and systematically.

---

# INPUT
You will receive:
- An image
- A question

---

# STEP 1 — TASK CLASSIFICATION

Classify the question into exactly ONE of the following categories:

- Depth
- Distance
- Relation
- Existence
- Count
- Instance_Location
- Orientation
- Size
- Reach

Rules:
- Select only one category.
- If multiple seem relevant, choose the most dominant spatial reasoning type required to answer correctly.
- Do not skip this step.

---

# STEP 2 — TASK-SPECIFIC PLAN

Based on the selected category:

1. Define the key spatial cues needed.
2. Identify relevant visual features (e.g., occlusion, perspective, alignment, relative scale).
3. Explain your strategy to solve this specific task.
4. Avoid superficial shortcuts or guessing.

---

# STEP 3 — STEP-BY-STEP REASONING

Follow a strict logical reasoning process:

- Analyze the image carefully.
- Extract relevant spatial information.
- Apply geometric or spatial logic when necessary.
- Ensure each reasoning step follows logically from the previous one.
- Do NOT jump directly to the answer.

---

# STEP 4 — FINAL ANSWER

Provide:
- A concise final answer.
- If multiple choices exist, clearly indicate the selected option.

---

# OUTPUT FORMAT (MANDATORY)

Task Category:
<One of the 9 categories>

Reasoning Plan:
<Brief task-specific plan>

Step-by-Step Reasoning:
<Logical reasoning steps>

Final Answer:
<Clear final answer>

---

# QUESTION

{question}
"""


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_runner(model_name: str, config: dict):
    models_cfg = config.get("models", {})
    if model_name == "qwen3_4b":
        m_cfg = models_cfg.get("qwen3_4b", {})
        if not m_cfg.get("enabled", True):
            return None
        eval_cfg = config.get("eval", {})
        return Qwen3Runner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device=m_cfg.get("device", "cuda"),
            use_flash_attn=eval_cfg.get("use_flash_attn", True),
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
            use_flash_attn=m_cfg.get("use_flash_attn", False),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _unload_model(runner):
    if runner is None:
        return
    if hasattr(runner, "model"):
        del runner.model
    if hasattr(runner, "processor"):
        del runner.processor
    if hasattr(runner, "tokenizer"):
        del runner.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model_on_benchmark(
    model_name: str,
    config: dict,
    dataset,
    benchmark: str,
    output_dir: Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
):
    """Run one model on 3DSRBench. Returns (accuracy, details_list)."""
    runner = get_runner(model_name, config)
    if runner is None:
        raise RuntimeError(f"Model not enabled: {model_name}")

    print(f"  [load] {model_name} ({getattr(runner, 'model_id', model_name)})")

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = model_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

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
            details.append({"idx": i, "error": "no_image", "gt": gt})
            continue

        full_prompt = SPATIAL_REASONING_PROMPT.format(question=query)

        try:
            response = runner.generate(
                image,
                full_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            letter = normalize_answer_only(response)
            preds.append(letter)
            gt_list.append(gt)
            details.append({
                "idx": i,
                "query": query,
                "gt": gt,
                "pred": letter,
                "category": category,
                "full_response": response,
            })

            # Sauvegarder réponse + GT + démarche par sample
            sample_path = responses_dir / f"sample_{i:05d}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("=== QUERY ===\n")
                f.write(query + "\n\n")
                f.write("=== GT ===\n")
                f.write(gt + "\n\n")
                f.write("=== FULL RESPONSE (démarche) ===\n")
                f.write(response + "\n\n")
                f.write("=== EXTRACTED PRED ===\n")
                f.write(letter + "\n")

        except Exception as e:
            preds.append("")
            gt_list.append(gt)
            details.append({"idx": i, "error": str(e), "gt": gt})

    acc = accuracy(preds, gt_list)
    _unload_model(runner)
    return acc, details


def main():
    parser = argparse.ArgumentParser(description="Single-agent eval on 3DSRBench: Qwen3, Sa2VA, LLaVA4D")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    output_dir = Path(config.get("output", {}).get("dir", "results"))
    benchmark = "3dsrbench"

    if eval_cfg.get("use_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / benchmark / "single_eval" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {benchmark}... (seed={args.seed})")
    dataset = load_benchmark(
        benchmark,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"  {len(dataset)} samples")

    models = ["qwen3_4b", "sa2va", "llava4d"]
    results_table = []

    for model_name in models:
        print(f"\n--- {model_name} ---")
        acc, details = run_model_on_benchmark(
            model_name,
            config,
            dataset,
            benchmark,
            run_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=eval_cfg.get("mas_temperature", 0.0),
        )
        results_table.append({"model": model_name, "accuracy": acc, "n": len(details)})

        with open(run_dir / model_name / "details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(run_dir / model_name / "results.json", "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "accuracy": acc, "n": len(details)}, f, indent=2)

        print(f"  Accuracy: {acc:.4f} ({len(details)} samples)")

    # Tableau récapitulatif
    summary_path = run_dir / "accuracy_table.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 3DSRBench — Single-Agent Accuracy\n\n")
        f.write("| Model | Accuracy | N |\n")
        f.write("|-------|----------|---|\n")
        for r in results_table:
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['n']} |\n")

    # Afficher le tableau
    print("\n" + "=" * 50)
    print("TABLEAU FINAL — Accuracy par modèle")
    print("=" * 50)
    print(f"{'Model':<15} {'Accuracy':<12} {'N':<8}")
    print("-" * 35)
    for r in results_table:
        print(f"{r['model']:<15} {r['accuracy']:.4f}       {r['n']:<8}")
    print("=" * 50)
    print(f"\nRésultats: {run_dir}")


if __name__ == "__main__":
    main()
