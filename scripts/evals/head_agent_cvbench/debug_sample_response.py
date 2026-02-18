#!/usr/bin/env python3
"""
Fetch 1 sample from each model and print raw response. For extraction debugging.
Usage: python scripts/evals/head_agent_cvbench/debug_sample_response.py [--model gpt5_2]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_category, get_benchmark_image
from prompt import build_category_routing_prompt, get_categories

# Import runners
_runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
_runners_spec = __import__("importlib").util.spec_from_file_location("runners", _runners_path)
_runners = __import__("importlib").util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)


def get_runner(model_key, config):
    cfg = config.get("models", {}).get(model_key, {})
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key:
        return None
    model_id = cfg.get("model_id", "")
    if model_key == "claude_opus_4_5":
        return _runners.ClaudeRunner(model_id=model_id, api_key=api_key)
    if model_key == "gpt5_2":
        return _runners.GPT4oRunner(model_id=model_id, api_key=api_key)
    if model_key == "glm5":
        base_url = cfg.get("base_url", "https://openrouter.ai/api/v1")
        return _runners.OpenRouterRunner(model_id=model_id, api_key=api_key, base_url=base_url)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt5_2", "glm5", "claude_opus_4_5"], default="gpt5_2")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = get_runner(args.model, config)
    if not runner:
        print(f"No runner for {args.model} (check API key)")
        return

    dataset = load_benchmark("cvbench", max_samples=1, seed=42)
    ex = dataset[0]
    query = get_benchmark_prompt(ex, "cvbench")
    gt = get_benchmark_category(ex, "cvbench")
    prompt = build_category_routing_prompt(query, "cvbench")
    image = get_benchmark_image(ex, "cvbench")
    if image is None:
        from PIL import Image
        image = Image.new("RGB", (1, 1), color="white")

    print(f"=== {args.model} raw response (GT={gt}) ===\n")
    resp = runner.generate(image, prompt, temperature=0.0, max_tokens=512)
    print(repr(resp))
    print("\n--- full text ---")
    print(resp)


if __name__ == "__main__":
    main()
