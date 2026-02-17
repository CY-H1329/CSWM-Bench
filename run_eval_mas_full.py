#!/usr/bin/env python3
"""
MAS 전체 평가: Qwen3 X3, Sa2VA X3, LLaVA4D X3
- 전체 데이터 사용 (max_samples 없음)
- category별 정확도 정리
- 각 단계(Head, Perception, Reasoning) 결과를 text로 저장

Usage:
  python run_eval_mas_full.py --benchmark 3dsrbench
  python run_eval_mas_full.py --benchmark 3dsrbench --seed 42
"""
import argparse
import gc
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
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


# 평가할 3가지 조합: 각 모델을 Head/Perception/Reasoning 모두에 사용
COMBINATIONS = [
    ("qwen3_4b", "qwen3_4b", "qwen3_4b"),
    ("sa2va", "sa2va", "sa2va"),
    ("llava4d", "llava4d", "llava4d"),
]


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
        eval_cfg = config.get("eval", {})
        return Qwen3Runner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device=m_cfg.get("device", "cuda"),
            use_flash_attn=eval_cfg.get("use_flash_attn", True),
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
            use_flash_attn=m_cfg.get("use_flash_attn", False),
        )
    elif model_name == "gpt":
        m_cfg = config.get("models", {}).get("gpt", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            return None
        return GPTRunner(model_id=m_cfg.get("model_id", "gpt-4o"), api_key=api_key)
    elif model_name == "gemini":
        m_cfg = config.get("models", {}).get("gemini", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        return GeminiRunner(model_id=m_cfg.get("model_id", "gemini-2.0-flash"), api_key=api_key)
    raise ValueError(f"Unknown model: {model_name}")


def make_generate_fn(runner, eval_cfg: dict, use_mas_temperature: bool = True):
    temp = eval_cfg.get("mas_temperature", eval_cfg.get("temperature", 0.0)) if use_mas_temperature else eval_cfg.get("temperature", 0.0)
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


def _unload_model(runner):
    """Explicitly unload model to free GPU memory and avoid cross-contamination."""
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


def run_single_combination(
    args,
    config,
    head_name,
    perc_name,
    reas_name,
    dataset,
    run_dir,
):
    """Run MAS for one combination."""
    eval_cfg = config.get("eval", {})
    runners_to_unload = []
    if head_name == perc_name == reas_name:
        runner = get_runner(head_name, config)
        if runner is None:
            raise RuntimeError(f"Model not enabled: {head_name}")
        runners_to_unload = [runner]
        model_id_used = getattr(runner, "model_id", head_name)
        print(f"  [load] {model_id_used}")
        gen_fn = make_generate_fn(runner, eval_cfg)
        head_gen = perc_gen = reas_gen = gen_fn
    else:
        head_runner = get_runner(head_name, config)
        perc_runner = get_runner(perc_name, config)
        reas_runner = get_runner(reas_name, config)
        if not all([head_runner, perc_runner, reas_runner]):
            raise RuntimeError(f"All agents must be enabled: {head_name}, {perc_name}, {reas_name}")
        runners_to_unload = [head_runner, perc_runner, reas_runner]
        model_id_used = f"{getattr(head_runner,'model_id',head_name)} / {getattr(perc_runner,'model_id',perc_name)} / {getattr(reas_runner,'model_id',reas_name)}"
        print(f"  [load] {model_id_used}")
        head_gen = make_generate_fn(head_runner, eval_cfg)
        perc_gen = make_generate_fn(perc_runner, eval_cfg)
        reas_gen = make_generate_fn(reas_runner, eval_cfg)

    preds = []
    gt_list = []
    details = []
    by_category = defaultdict(lambda: {"preds": [], "gts": []})

    step_outputs_dir = run_dir / "step_outputs"
    step_outputs_dir.mkdir(parents=True, exist_ok=True)

    # Verification: run 1 sample and log head_output to verify model is correct
    _first_head_out = None
    for i in tqdm(range(len(dataset)), desc=f"{head_name}_{perc_name}_{reas_name}"):
        example = dataset[i]
        image = get_benchmark_image(example, args.benchmark)
        query = get_benchmark_prompt(example, args.benchmark)
        gt = get_benchmark_answer(example, args.benchmark)
        category = get_benchmark_category(example, args.benchmark) or "unknown"

        if image is None:
            preds.append("")
            gt_list.append(gt)
            details.append({"idx": i, "error": "no_image"})
            by_category[category]["preds"].append("")
            by_category[category]["gts"].append(gt)
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
            if args.benchmark in ("cvbench", "3dsrbench") and len(gt) == 1 and gt in "ABCDEF":
                letter = normalize_answer_only(final)
            else:
                letter = final.strip()
                if "answer:" in letter.lower():
                    letter = letter.split("answer:")[-1].strip()
                if "\n" in letter:
                    letter = letter.split("\n")[-1].strip()

            preds.append(letter)
            gt_list.append(gt)
            by_category[category]["preds"].append(letter)
            by_category[category]["gts"].append(gt)

            if _first_head_out is None:
                _first_head_out = (out.get("head_output", ""), final[:80])
            details.append({
                "idx": i,
                "task_class": out["task_class"],
                "head_output": out.get("head_output", ""),
                "perception_output": out["perception_output"],
                "reasoning_output": final,
                "pred": letter,
                "gt": gt,
                "category": category,
            })

            # 각 단계 결과를 text 파일로 저장
            step_path = step_outputs_dir / f"sample_{i:05d}.txt"
            with open(step_path, "w", encoding="utf-8") as f:
                f.write("=== QUERY ===\n")
                f.write(query + "\n\n")
                f.write("=== HEAD (Task Classification) ===\n")
                f.write(out.get("head_output", "") + "\n\n")
                f.write("=== PERCEPTION (Extracted Info) ===\n")
                f.write(out["perception_output"] + "\n\n")
                f.write("=== REASONING (Final Answer) ===\n")
                f.write(final + "\n\n")
                f.write("=== PRED ===\n")
                f.write(letter + "\n\n")
                f.write("=== GT ===\n")
                f.write(gt + "\n")

        except Exception as e:
            preds.append("")
            gt_list.append(gt)
            details.append({"idx": i, "error": str(e)})
            by_category[category]["preds"].append("")
            by_category[category]["gts"].append(gt)

    # Compute accuracy
    if args.benchmark in ("cvbench", "3dsrbench"):
        acc = accuracy(preds, gt_list)
    else:
        def _norm(s):
            return str(s).strip().lower()
        acc = sum(_norm(p) == _norm(g) for p, g in zip(preds, gt_list)) / len(gt_list) if gt_list else 0.0

    # Per-category accuracy
    def _norm(s):
        return str(s).strip().lower()

    by_category_acc = {}
    for cat, d in sorted(by_category.items()):
        if not d["gts"]:
            by_category_acc[cat] = {"accuracy": 0.0, "total": 0, "correct": 0}
            continue
        if args.benchmark in ("cvbench", "3dsrbench"):
            correct = sum(1 for p, g in zip(d["preds"], d["gts"]) if p == g)
        else:
            correct = sum(1 for p, g in zip(d["preds"], d["gts"]) if _norm(p) == _norm(g))
        by_category_acc[cat] = {
            "accuracy": correct / len(d["gts"]),
            "total": len(d["gts"]),
            "correct": correct,
        }

    # Verification: print first sample's head_output to verify model differs per combination
    if _first_head_out is not None:
        print(f"  [verify] sample0 head_out={repr(_first_head_out[0][:60])}... pred={repr(_first_head_out[1][:40])}...")

    # Explicitly unload model before next combination (avoid GPU memory + cross-contamination)
    for r in runners_to_unload:
        _unload_model(r)
    del runners_to_unload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, details, by_category_acc, model_id_used


def main():
    parser = argparse.ArgumentParser(description="MAS full evaluation: Qwen3 X3, Sa2VA X3, LLaVA4D X3")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--benchmark", default="3dsrbench", choices=["omni3d", "cvbench", "3dsrbench"])
    parser.add_argument("--seed", type=int, default=None, help="Dataset seed (default: config eval.mas_seed)")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    output_dir = Path(config.get("output", {}).get("dir", "results"))

    if eval_cfg.get("use_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = args.seed if args.seed is not None else eval_cfg.get("mas_seed", 42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = output_dir / "runs" / args.benchmark / "full_eval" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset (no max_samples)
    print(f"Loading {args.benchmark}... (seed={seed}, mode=all)")
    dataset = load_benchmark(args.benchmark, seed=seed)
    print(f"  {len(dataset)} samples (full)")

    all_results = []

    for head_name, perc_name, reas_name in COMBINATIONS:
        run_name = f"{head_name}_{perc_name}_{reas_name}"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {run_name} ---")
        acc, details, by_category_acc, model_id_used = run_single_combination(
            args, config, head_name, perc_name, reas_name, dataset, run_dir
        )

        results = {
            "benchmark": args.benchmark,
            "combination": run_name,
            "model_id": model_id_used,
            "accuracy": acc,
            "num_samples": len(details),
            "sample_mode": "all",
            "seed": seed,
            "head": head_name,
            "perception": perc_name,
            "reasoning": reas_name,
            "by_category": by_category_acc,
        }

        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(run_dir / "details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # category별 요약 텍스트
        summary_path = run_dir / "by_category_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# {run_name} - Category Summary\n\n")
            f.write(f"Overall Accuracy: {acc:.4f} ({len(details)} samples)\n\n")
            f.write("| Category | Total | Correct | Accuracy |\n")
            f.write("|----------|-------|---------|----------|\n")
            for cat in sorted(by_category_acc.keys()):
                d = by_category_acc[cat]
                f.write(f"| {cat} | {d['total']} | {d['correct']} | {d['accuracy']:.4f} |\n")

        all_results.append(results)
        print(f"  Accuracy: {acc:.4f} ({len(details)} samples)")
        print(f"  Results: {run_dir}")

    # 전체 요약
    summary_path = base_dir / "all_combinations_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# MAS Full Evaluation Summary\n\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("| Combination | Accuracy |\n")
        f.write("|-------------|----------|\n")
        for r in all_results:
            f.write(f"| {r['combination']} | {r['accuracy']:.4f} |\n")

        f.write("\n## By Category\n\n")
        all_cats = sorted({c for r in all_results for c in r.get("by_category", {})})
        if all_cats:
            f.write("| Category | " + " | ".join(r["combination"] for r in all_results) + " |\n")
            f.write("|" + "----------|" * (len(all_results) + 1) + "\n")
            for cat in all_cats:
                row = [cat]
                for r in all_results:
                    bc = r.get("by_category", {}).get(cat, {})
                    row.append(f"{bc.get('accuracy', 0):.4f}" if bc else "-")
                f.write("| " + " | ".join(row) + " |\n")

    with open(base_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\n--- All done ---")
    print(f"Results: {base_dir}")


if __name__ == "__main__":
    main()
