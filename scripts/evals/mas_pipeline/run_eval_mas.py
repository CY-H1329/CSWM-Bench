#!/usr/bin/env python3
"""
Spatial MAS Pipeline Evaluation — Head → 3 Specialists → Reasoning → Score Update.

Runs the full pipeline on CV-Bench (or 3DSRBench):
1. Head-Agent (GPT-5.2) infers category, selects 3 agents, creates coordination policy
2. 3 Specialist agents solve the task (CoT + answer)
3. Reasoning Agent (DeepSeek-VL) synthesizes final answer
4. Per-agent scores updated based on correctness

Usage:
  python scripts/evals/mas_pipeline/run_eval_mas.py --test
  python scripts/evals/mas_pipeline/run_eval_mas.py --max_samples 50
  python scripts/evals/mas_pipeline/run_eval_mas.py --full_dataset

Env: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm
from PIL import Image

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
)
from src.agents.mas import run_spatial_mas_pipeline, ScoreManager
from src.agents.mas.config import CVBENCH_TO_UNIFIED, TASK_CATEGORIES

# API Runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
GPT4oRunner = _runners.GPT4oRunner
ClaudeRunner = _runners.ClaudeRunner
GeminiRunner = _runners.GeminiRunner
DeepSeekVLRunner = _runners.DeepSeekVLRunner
OpenRouterRunner = _runners.OpenRouterRunner

# GPU Runners (H100)
try:
    from src.models.qwen3 import Qwen3Runner
    from src.models.sa2va import Sa2VARunner
    from src.models.llava import LLaVARunner
    from src.models.deepseek_vl import DeepSeekVLRunner as DeepSeekVLGPURunner
    GPU_AVAILABLE = True
except ImportError:
    Qwen3Runner = Sa2VARunner = LLaVARunner = DeepSeekVLGPURunner = None
    GPU_AVAILABLE = False


def _norm_answer(s: str) -> str:
    """Normalize answer to (A)/(B)/(C)/(D)."""
    s = (s or "").strip().upper()
    for c in "ABCD":
        if c in s or f"({c})" in s:
            return f"({c})"
    return s


def build_runners(config: dict):
    """Build Head, Specialist, and Reasoning runners from config.
    Specialists: GPU (qwen3_4b, sa2va, llava4d) on H100, API for claude/gpt4o/gemini.
    """
    head_cfg = config.get("head_agent", {})
    head_key = os.environ.get(head_cfg.get("api_key_env", ""), "").strip()
    head_runner = GPT4oRunner(
        model_id=head_cfg.get("model_id", "gpt-4o"),
        api_key=head_key,
    ) if head_key else None

    reason_cfg = config.get("reasoning_agent", {})
    reason_runner = None
    if reason_cfg.get("runner") == "gpu" and GPU_AVAILABLE and DeepSeekVLGPURunner:
        try:
            reason_runner = DeepSeekVLGPURunner(
                model_id=reason_cfg.get("model_id", "deepseek-community/deepseek-vl-7b-chat"),
                device=reason_cfg.get("device", "cuda"),
            )
        except Exception as e:
            print(f"[skip] Reasoning GPU (DeepSeek-VL): {e}")
    else:
        reason_key = os.environ.get(reason_cfg.get("api_key_env", ""), "").strip()
        if reason_key:
            reason_runner = DeepSeekVLRunner(
                model_id=reason_cfg.get("model_id", "deepseek-vl"),
                api_key=reason_key,
                base_url=reason_cfg.get("base_url", "https://api.deepseek.com"),
            )

    specialists_cfg = config.get("specialists", {})
    specialist_runners = {}
    for name, cfg in specialists_cfg.items():
        runner_type = cfg.get("runner", "api")
        model_id = cfg.get("model_id", "")

        if runner_type == "gpu" and GPU_AVAILABLE:
            device = cfg.get("device", "cuda")
            try:
                if name == "qwen3_4b":
                    specialist_runners[name] = Qwen3Runner(model_id=model_id, device=device)
                elif name == "sa2va":
                    specialist_runners[name] = Sa2VARunner(model_id=model_id, device=device)
                elif name == "llava4d":
                    specialist_runners[name] = LLaVARunner(model_id=model_id, device=device)
                else:
                    specialist_runners[name] = None
            except Exception as e:
                print(f"[skip] {name} GPU: {e}")
                specialist_runners[name] = None
        elif runner_type == "api":
            api_runner = cfg.get("api_runner", "")
            key = os.environ.get(cfg.get("api_key_env", ""), "").strip()
            if not key:
                specialist_runners[name] = None
                continue
            if api_runner == "claude":
                specialist_runners[name] = ClaudeRunner(model_id=model_id, api_key=key)
            elif api_runner == "openai":
                specialist_runners[name] = GPT4oRunner(model_id=model_id, api_key=key)
            elif api_runner == "gemini":
                specialist_runners[name] = GeminiRunner(model_id=model_id, api_key=key)
            else:
                specialist_runners[name] = None
        else:
            specialist_runners[name] = None

    return head_runner, specialist_runners, reason_runner


def main():
    parser = argparse.ArgumentParser(description="Spatial MAS Pipeline Eval")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_mas.yaml"))
    parser.add_argument("--test", action="store_true", help="5 samples only")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--full_dataset", action="store_true")
    parser.add_argument("--benchmark", choices=["cvbench", "3dsrbench"], default="cvbench")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    head_runner, specialist_runners, reason_runner = build_runners(config)
    if not head_runner:
        print("ERROR: Head runner (OPENAI_API_KEY) required.")
        sys.exit(1)
    if not reason_runner:
        print("ERROR: Reasoning runner required. Set runner: gpu in config or DEEPSEEK_API_KEY for API.")
        sys.exit(1)

    ds_cfg = config.get("dataset", {})
    max_samples = ds_cfg.get("test_samples", 5) if args.test else (
        None if args.full_dataset else (args.max_samples or ds_cfg.get("max_samples", 50))
    )
    benchmark = args.benchmark or ds_cfg.get("benchmark", "cvbench")
    seed = args.seed or ds_cfg.get("seed", 42)

    ds = load_benchmark(benchmark, max_samples=max_samples, seed=seed)

    out_dir = Path(config.get("output", {}).get("dir", "results/runs/mas_pipeline"))
    subdir = "test" if args.test else ("full" if args.full_dataset else datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(out_dir) / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    def head_gen(img: Image.Image, prompt: str) -> str:
        return head_runner.generate(img, prompt, max_tokens=2048)

    def spec_gen(agent_name: str, img: Image.Image, prompt: str) -> str:
        r = specialist_runners.get(agent_name)
        if not r:
            return ""
        # GPU runners use max_new_tokens, API use max_tokens
        mod = type(r).__module__ or ""
        if "src.models" in mod:
            return r.generate(img, prompt, max_new_tokens=2048)
        return r.generate(img, prompt, max_tokens=2048)

    def reason_gen(img: Image.Image, prompt: str) -> str:
        mod = type(reason_runner).__module__ or ""
        if "src.models" in mod:
            return reason_runner.generate(img, prompt, max_new_tokens=1024)
        return reason_runner.generate(img, prompt, max_tokens=1024)

    score_manager = ScoreManager()
    category_seen = {c: False for c in TASK_CATEGORIES}
    score_history = [score_manager.to_dict()]

    results = []
    correct = 0
    total = 0

    for i in tqdm(range(len(ds)), desc="MAS Pipeline"):
        ex = ds[i]
        img = get_benchmark_image(ex, benchmark)
        if img is None:
            continue
        query = get_benchmark_prompt(ex, benchmark)
        gt = get_benchmark_answer(ex, benchmark)
        gt_norm = _norm_answer(gt)

        out = run_spatial_mas_pipeline(
            image=img,
            query=query,
            gt_answer=gt,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            score_manager=score_manager,
            category_seen=category_seen,
        )

        if "error" in out:
            results.append({"idx": i, "error": out["error"], "gt": gt})
            continue

        pred = out.get("final_answer", "")
        pred_norm = _norm_answer(pred)
        is_correct = pred_norm == gt_norm
        if is_correct:
            correct += 1
        total += 1

        cat = out.get("predicted_category", "")
        category_seen[cat] = True

        agent_results = [
            {k: v for k, v in r.items() if k != "raw"}
            for r in out.get("agent_results", [])
        ]
        score_history.append(score_manager.to_dict())
        results.append({
            "idx": i,
            "predicted_category": cat,
            "selected_agents": out.get("selected_agents", []),
            "final_answer": pred,
            "gt": gt,
            "correct": is_correct,
            "agent_results": agent_results,
            "reasoning_justification": out.get("reasoning_justification", ""),
            "score_table_after_turn": score_manager.to_dict(),
        })

        if (i + 1) % 10 == 0:
            with open(run_dir / "progress.json", "w") as f:
                json.dump({
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else 0,
                    "score_table": score_manager.to_dict(),
                }, f, indent=2)

    acc = correct / total if total else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.2%}")

    summary = {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "benchmark": benchmark,
        "max_samples": max_samples,
        "score_table": score_manager.to_dict(),
        "score_history": score_history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
