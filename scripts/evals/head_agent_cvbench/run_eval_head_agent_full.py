#!/usr/bin/env python3
"""
Head-Agent 5가지 핵심 능력 평가.

1. Task Decomposition - 문제 정확 분류
2. Routing Decision - 어떤 agent를 고를지
3. Complexity Estimation - 간단 vs 복잡 판단
4. Strategy Planning - tool/strategy 초안
5. Trust-Aware Logging - reasoning trace 구조화

Usage:
  python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --max_samples 50
  python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --capability routing --model gpt5_2
  python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --capability all --benchmark cvbench

Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_category, get_benchmark_image
from prompt import get_categories, build_category_routing_prompt
from prompts_head_agent import (
    build_task_decomposition_prompt,
    build_routing_decision_prompt,
    build_complexity_estimation_prompt,
    build_strategy_planning_prompt,
    build_trust_aware_logging_prompt,
    ROUTING_OPTIONS,
)

# Runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
ClaudeRunner = _runners.ClaudeRunner
GPT4oRunner = _runners.GPT4oRunner
OpenRouterRunner = _runners.OpenRouterRunner

CAPABILITIES = ["task_decomposition", "routing", "complexity", "strategy", "trust_logging"]


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
        return OpenRouterRunner(model_id=model_id, api_key=api_key, base_url=base_url, text_only=True)
    return None


def _normalize(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower().replace(" ", "").replace("_", "").replace("-", ""))


# === Extraction & Evaluation ===

def extract_category(response: str, categories: list) -> str:
    """Extract category from response."""
    if not response or not response.strip():
        return ""
    text = response.strip()
    cat_norm = {_normalize(c): c for c in categories}
    for pat in [
        r"Category\s*:\s*([A-Za-z][A-Za-z0-9_\s\-]*?)(?:\n|\.|$)",
        r"[Cc]ategory\s+is\s+([A-Za-z][A-Za-z0-9_\s\-]*?)(?:\n|\.|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = re.sub(r"[^\w\s\-]", "", m.group(1).strip()).strip()
            raw_norm = _normalize(raw)
            for cn, c in cat_norm.items():
                if raw_norm == cn or cn in raw_norm or raw_norm in cn:
                    return c
            for cn, c in cat_norm.items():
                if raw.lower() in c.lower() or c.lower() in raw.lower():
                    return c
            return raw
    for c in categories:
        if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
            return c
    return ""


def extract_route(response: str) -> str:
    """Extract routing decision."""
    if not response:
        return ""
    text = response.strip()
    for opt in ROUTING_OPTIONS:
        if re.search(rf"\b{re.escape(opt)}\b", text):
            return opt
    m = re.search(r"Route\s*:\s*(\w+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_complexity(response: str) -> str:
    """Extract complexity 1-5."""
    if not response:
        return ""
    text = response.strip()
    m = re.search(r"Complexity\s*:\s*([1-5])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.search(r"\b[1-5]\b", text):
        for i in "12345":
            if re.search(rf"\b{i}\b", text):
                return i
    return ""


def eval_strategy_format(response: str) -> dict:
    """Check if strategy has structure (numbered steps, keywords)."""
    if not response:
        return {"valid": False, "has_steps": False, "has_keywords": False}
    text = response.lower()
    keywords = ["depth", "count", "local", "relation", "object", "spatial", "extract", "detect", "compare"]
    has_steps = bool(re.search(r"\d+\.\s*\w+", text))
    has_keywords = any(kw in text for kw in keywords)
    return {"valid": len(response) > 20, "has_steps": has_steps, "has_keywords": has_keywords}


def eval_trust_logging(response: str) -> dict:
    """Check if output is valid structured JSON with required keys."""
    if not response:
        return {"valid_json": False, "has_keys": False, "keys": []}
    text = response.strip()
    # Extract JSON block if wrapped in markdown
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    elif "{" in text:
        start = text.index("{")
        depth, end = 0, start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        text = text[start : end + 1]
    try:
        j = json.loads(text)
        if not isinstance(j, dict):
            return {"valid_json": False, "has_keys": False, "keys": []}
        required = ["reasoning", "category", "route", "complexity", "confidence"]
        found = [k for k in required if k in j]
        return {"valid_json": True, "has_keys": len(found) >= 3, "keys": list(j.keys())}
    except json.JSONDecodeError:
        return {"valid_json": False, "has_keys": False, "keys": []}


def main():
    parser = argparse.ArgumentParser(description="Head-Agent 5가지 능력 평가")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--capability", choices=CAPABILITIES + ["all"], default="all")
    parser.add_argument("--model", choices=["gpt5_2", "claude_opus_4_5", "glm5"])
    parser.add_argument("--benchmark", choices=["cvbench", "3dsrbench", "all"], default="cvbench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output", {}).get("dir", "results"))
    subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_keys = [args.model] if args.model else ["gpt5_2", "claude_opus_4_5", "glm5"]
    caps = CAPABILITIES if args.capability == "all" else [args.capability]
    benchmarks = ["cvbench", "3dsrbench"] if args.benchmark == "all" else [args.benchmark]

    for benchmark in benchmarks:
        dataset = load_benchmark(benchmark, max_samples=args.max_samples, seed=args.seed)
        categories = get_categories(benchmark)
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {benchmark.upper()} | {len(dataset)} samples")
        print("=" * 60)

        for cap in caps:
            run_dir = output_dir / "runs" / "head_agent" / benchmark / cap / subdir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "responses").mkdir(exist_ok=True)

            print(f"\n--- Capability: {cap} ---")
            _run_capability(
                capability=cap,
                benchmark=benchmark,
                dataset=dataset,
                categories=categories,
                model_keys=model_keys,
                config=config,
                run_dir=run_dir,
                args=args,
            )


def _run_capability(capability, benchmark, dataset, categories, model_keys, config, run_dir, args):
    """Run one capability evaluation."""
    results_table = []
    for model_key in model_keys:
        runner = get_runner(model_key, config)
        if not runner:
            print(f"  [skip] {model_key}")
            continue

        details = []
        for i in tqdm(range(len(dataset)), desc=model_key):
            ex = dataset[i]
            query = get_benchmark_prompt(ex, benchmark)
            gt_cat = get_benchmark_category(ex, benchmark)
            gt_cat_norm = gt_cat
            for c in categories:
                if c.lower() == (gt_cat or "").lower():
                    gt_cat_norm = c
                    break

            # Build prompt per capability
            if capability == "task_decomposition":
                prompt = build_task_decomposition_prompt(query, benchmark)
            elif capability == "routing":
                prompt = build_routing_decision_prompt(query, gt_cat_norm or "Unknown", benchmark)
            elif capability == "complexity":
                prompt = build_complexity_estimation_prompt(query, gt_cat_norm or "Unknown")
            elif capability == "strategy":
                prompt = build_strategy_planning_prompt(query, gt_cat_norm or "Unknown", "Perception")
            elif capability == "trust_logging":
                prompt = build_trust_aware_logging_prompt(
                    query, gt_cat_norm or "Unknown", "Perception", "3"
                )
            else:
                prompt = build_task_decomposition_prompt(query, benchmark)

            try:
                from src.benchmarks import get_benchmark_image
                image = get_benchmark_image(ex, benchmark)
                if image is None or (model_key == "glm5"):
                    from PIL import Image
                    image = Image.new("RGB", (1, 1), color="white")
                response = runner.generate(image, prompt, temperature=0.0, max_tokens=args.max_tokens)
            except Exception as e:
                details.append({"idx": i, "error": str(e), "response": ""})
                continue

            # Evaluate
            pred = None
            if capability == "task_decomposition":
                pred = extract_category(response, categories)
                correct = pred and gt_cat_norm and pred.lower() == gt_cat_norm.lower()
                score = 1.0 if correct else 0.0
            elif capability == "routing":
                pred = extract_route(response)
                valid = pred in ROUTING_OPTIONS
                score = 1.0 if valid else 0.0
            elif capability == "complexity":
                pred = extract_complexity(response)
                valid = pred in "12345"
                score = 1.0 if valid else 0.0
            elif capability == "strategy":
                ev = eval_strategy_format(response)
                score = (0.5 if ev["has_steps"] else 0) + (0.5 if ev["has_keywords"] else 0)
            elif capability == "trust_logging":
                ev = eval_trust_logging(response)
                score = (0.5 if ev["valid_json"] else 0) + (0.5 if ev["has_keys"] else 0)
            else:
                score = 0.0

            details.append({
                "idx": i,
                "query": query[:200],
                "gt_category": gt_cat_norm,
                "response": response,
                "score": score,
                "pred": pred,
            })

        avg_score = sum(d["score"] for d in details) / len(details) if details else 0
        results_table.append({"model": model_key, "score": avg_score, "n": len(details)})

        model_dir = run_dir / model_key
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / "details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(model_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"model": model_key, "capability": capability, "score": avg_score, "n": len(details)}, f, indent=2)

        print(f"  {model_key}: {avg_score:.4f}")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Head-Agent {capability} — {benchmark.upper()}\n\n")
        f.write("| Model | Score | N |\n|-------|-------|---|\n")
        for r in results_table:
            f.write(f"| {r['model']} | {r['score']:.4f} | {r['n']} |\n")

    print(f"  Results: {run_dir}")


if __name__ == "__main__":
    main()
