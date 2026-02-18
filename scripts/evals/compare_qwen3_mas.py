#!/usr/bin/env python3
"""
Compare Qwen3-4B (best single model) vs MAS on 10 samples from multi_object_viewpoint_towards_object.

Qwen3-4B performs worst on this category (0.309 acc). We run both on the same 10 samples
and compare results.

Usage:
  python scripts/evals/compare_qwen3_mas.py
  python scripts/evals/compare_qwen3_mas.py --num_samples 10 --seed 42
  python scripts/evals/compare_qwen3_mas.py --qwen_only   # Verify Qwen first, skip MAS
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
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
from src.data import normalize_answer_only, accuracy
from src.agents.mas import run_spatial_mas_pipeline, ScoreManager
from src.agents.mas.config import TASK_CATEGORIES

# API Runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
GPT4oRunner = _runners.GPT4oRunner
ClaudeRunner = _runners.ClaudeRunner
GeminiRunner = _runners.GeminiRunner
DeepSeekVLRunner = _runners.DeepSeekVLRunner

try:
    from src.models.qwen3 import Qwen3Runner
    from src.models.sa2va import Sa2VARunner
    from src.models.llava import LLaVARunner
    from src.models.deepseek_vl import DeepSeekVLRunner as DeepSeekVLGPURunner
    GPU_AVAILABLE = True
except ImportError:
    Qwen3Runner = Sa2VARunner = LLaVARunner = DeepSeekVLGPURunner = None
    GPU_AVAILABLE = False

TARGET_CATEGORY = "multi_object_viewpoint_towards_object"

_3DSRBENCH_CATS = [
    "location_above", "height_higher", "location_closer_to_camera",
    "multi_object_closer_to", "orientation_on_the_left", "multi_object_facing",
    "multi_object_same_direction", "orientation_in_front_of",
    "multi_object_viewpoint_towards_object", "orientation_viewpoint",
    "location_next_to", "multi_object_parallel",
]


def _build_qwen3_prompt(question: str) -> str:
    """Same prompt as run_eval_single_3dsrbench for fair comparison."""
    cats = "\n".join(f"- {c}" for c in _3DSRBENCH_CATS)
    return f"""# ROLE
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

{cats}

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
<One of the 12 categories>

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


def _norm_answer(s: str) -> str:
    s = (s or "").strip().upper()
    for c in "ABCD":
        if c in s or f"({c})" in s:
            return f"({c})"
    return s


def _canonical_letter(s: str) -> str:
    """Extract A/B/C/D for comparison. Handles both 'A' and '(A)' formats."""
    if not s:
        return ""
    s = str(s).strip().upper()
    if s in "ABCD":
        return s
    m = re.search(r"\(([A-D])\)", s)
    return m.group(1) if m else ""


def build_mas_runners(config: dict):
    """Build Head, Specialist, Reasoning runners."""
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
            print(f"[skip] Reasoning GPU: {e}")
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
            key = os.environ.get(cfg.get("api_key_env", ""), "").strip()
            if not key:
                specialist_runners[name] = None
                continue
            if cfg.get("api_runner") == "claude":
                specialist_runners[name] = ClaudeRunner(model_id=model_id, api_key=key)
            elif cfg.get("api_runner") == "openai":
                specialist_runners[name] = GPT4oRunner(model_id=model_id, api_key=key)
            elif cfg.get("api_runner") == "gemini":
                specialist_runners[name] = GeminiRunner(model_id=model_id, api_key=key)
            else:
                specialist_runners[name] = None
        else:
            specialist_runners[name] = None

    return head_runner, specialist_runners, reason_runner


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3-4B vs MAS on multi_object_viewpoint_towards_object")
    parser.add_argument("--config", default=str(ROOT / "scripts/evals/mas_pipeline/config_mas.yaml"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None, help="Output directory (default: results/comparison_qwen3_mas)")
    parser.add_argument("--qwen_only", action="store_true", help="Run only Qwen3-4B to verify it works (no MAS)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load 3DSRBench filtered to multi_object_viewpoint_towards_object
    print(f"Loading 3DSRBench (category={TARGET_CATEGORY})...")
    ds_full = load_benchmark(
        "3dsrbench",
        category_filter=[TARGET_CATEGORY],
        seed=args.seed,
    )
    n_total = len(ds_full)
    n_take = min(args.num_samples, n_total)
    indices = list(range(n_take))  # first N for reproducibility
    ds = ds_full.select(indices)
    print(f"  Using {n_take} samples (total in category: {n_total})")

    out_dir = Path(args.output_dir or "results/comparison_qwen3_mas")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Run Qwen3-4B
    print("\n--- Qwen3-4B (single model) ---")
    if not GPU_AVAILABLE or not Qwen3Runner:
        print("ERROR: Qwen3Runner not available (GPU/transformers)")
        qwen_results = [{"idx": i, "error": "no_runner", "gt": get_benchmark_answer(ds[i], "3dsrbench")} for i in range(len(ds))]
    else:
        qwen_runner = Qwen3Runner(
            model_id=config.get("specialists", {}).get("qwen3_4b", {}).get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device="cuda",
        )
        qwen_results = []
        for i in tqdm(range(len(ds)), desc="Qwen3-4B"):
            ex = ds[i]
            img = get_benchmark_image(ex, "3dsrbench")
            query = get_benchmark_prompt(ex, "3dsrbench")
            gt = get_benchmark_answer(ex, "3dsrbench")
            if img is None:
                qwen_results.append({"idx": i, "error": "no_image", "gt": gt, "pred": "", "correct": False})
                continue
            try:
                resp = qwen_runner.generate(img, _build_qwen3_prompt(query), max_new_tokens=1024)
                pred = normalize_answer_only(resp)
                correct = _canonical_letter(pred) == _canonical_letter(gt)
                qwen_results.append({
                    "idx": i, "query": query[:200], "gt": gt, "pred": pred,
                    "full_response": resp[:500], "correct": correct,
                })
            except Exception as e:
                qwen_results.append({"idx": i, "error": str(e), "gt": gt, "pred": "", "correct": False})

    qwen_correct = sum(1 for r in qwen_results if r.get("correct"))
    qwen_total = len(qwen_results)
    qwen_acc = qwen_correct / qwen_total if qwen_total else 0
    print(f"  Qwen3-4B: {qwen_correct}/{qwen_total} = {qwen_acc:.2%}")

    if args.qwen_only:
        print("\n[--qwen_only] Skipping MAS. Qwen verification complete.")
        print("\nPer-sample results:")
        print(f"{'Idx':<5} {'GT':<6} {'Qwen3':<8} {'Qwen✓':<6}")
        print("-" * 35)
        for i in range(len(ds)):
            qr = next((r for r in qwen_results if r.get("idx") == i), {})
            q_pred = qr.get("pred", "err")
            q_ok = "✓" if qr.get("correct") else "✗"
            gt = qr.get("gt", "?")
            print(f"{i:<5} {gt:<6} {q_pred:<8} {q_ok:<6}")
        with open(run_dir / "summary.json", "w") as f:
            json.dump({
                "category": TARGET_CATEGORY,
                "num_samples": n_take,
                "seed": args.seed,
                "qwen3_4b": {"correct": qwen_correct, "total": qwen_total, "accuracy": qwen_acc},
            }, f, indent=2)
        with open(run_dir / "qwen3_results.jsonl", "w") as f:
            for r in qwen_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {run_dir}")
        return

    # 3. Run MAS
    print("\n--- MAS Pipeline ---")
    head_runner, specialist_runners, reason_runner = build_mas_runners(config)
    if not head_runner or not reason_runner:
        print("ERROR: Head or Reasoning runner required (OPENAI_API_KEY, etc.)")
        mas_results = [{"idx": i, "error": "no_runner", "gt": get_benchmark_answer(ds[i], "3dsrbench"), "correct": False} for i in range(len(ds))]
        mas_correct, mas_total = 0, len(ds)
    else:
        def head_gen(img, prompt):
            return head_runner.generate(img, prompt, max_tokens=2048)

        def spec_gen(agent_name, img, prompt):
            r = specialist_runners.get(agent_name)
            if not r:
                return ""
            mod = type(r).__module__ or ""
            if "src.models" in mod:
                return r.generate(img, prompt, max_new_tokens=2048)
            return r.generate(img, prompt, max_tokens=2048)

        def reason_gen(img, prompt):
            mod = type(reason_runner).__module__ or ""
            if "src.models" in mod:
                return reason_runner.generate(img, prompt, max_new_tokens=1024)
            return reason_runner.generate(img, prompt, max_tokens=1024)

        score_manager = ScoreManager()
        category_seen = {c: False for c in TASK_CATEGORIES}
        mas_results = []

        for i in tqdm(range(len(ds)), desc="MAS"):
            ex = ds[i]
            img = get_benchmark_image(ex, "3dsrbench")
            query = get_benchmark_prompt(ex, "3dsrbench")
            gt = get_benchmark_answer(ex, "3dsrbench")
            gt_norm = _norm_answer(gt)
            if img is None:
                mas_results.append({"idx": i, "error": "no_image", "gt": gt, "correct": False})
                continue
            try:
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
                    mas_results.append({"idx": i, "error": out["error"], "gt": gt, "correct": False})
                    continue
                pred = out.get("final_answer", "")
                pred_norm = _norm_answer(pred)
                correct = _canonical_letter(pred) == _canonical_letter(gt)
                cat = out.get("predicted_category", "")
                category_seen[cat] = True
                mas_results.append({
                    "idx": i, "gt": gt, "pred": pred_norm, "correct": correct,
                    "predicted_category": cat, "selected_agents": out.get("selected_agents", []),
                })
            except Exception as e:
                mas_results.append({"idx": i, "error": str(e), "gt": gt, "correct": False})

        mas_correct = sum(1 for r in mas_results if r.get("correct"))
        mas_total = len(mas_results)

    mas_acc = mas_correct / mas_total if mas_total else 0
    print(f"  MAS: {mas_correct}/{mas_total} = {mas_acc:.2%}")

    # 4. Comparison report
    print("\n" + "=" * 60)
    print("COMPARISON: Qwen3-4B vs MAS")
    print(f"Category: {TARGET_CATEGORY} ({n_take} samples)")
    print("=" * 60)
    print(f"  Qwen3-4B:  {qwen_correct}/{qwen_total} = {qwen_acc:.2%}")
    print(f"  MAS:       {mas_correct}/{mas_total} = {mas_acc:.2%}")
    print(f"  Delta:     MAS {'+' if mas_acc >= qwen_acc else ''}{(mas_acc - qwen_acc):.2%}")
    print("=" * 60)

    # Per-sample table
    print("\nPer-sample results:")
    print(f"{'Idx':<5} {'GT':<6} {'Qwen3':<8} {'MAS':<8} {'Qwen✓':<6} {'MAS✓':<6}")
    print("-" * 50)
    for i in range(len(ds)):
        qr = next((r for r in qwen_results if r.get("idx") == i), {})
        mr = next((r for r in mas_results if r.get("idx") == i), {})
        q_pred = qr.get("pred", "err")
        m_pred = mr.get("pred", "err")
        q_ok = "✓" if qr.get("correct") else "✗"
        m_ok = "✓" if mr.get("correct") else "✗"
        gt = qr.get("gt", mr.get("gt", "?"))
        print(f"{i:<5} {gt:<6} {q_pred:<8} {m_pred:<8} {q_ok:<6} {m_ok:<6}")

    # Save
    summary = {
        "category": TARGET_CATEGORY,
        "num_samples": n_take,
        "seed": args.seed,
        "qwen3_4b": {"correct": qwen_correct, "total": qwen_total, "accuracy": qwen_acc},
        "mas": {"correct": mas_correct, "total": mas_total, "accuracy": mas_acc},
        "delta": mas_acc - qwen_acc,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "qwen3_results.jsonl", "w") as f:
        for r in qwen_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(run_dir / "mas_results.jsonl", "w") as f:
        for r in mas_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
