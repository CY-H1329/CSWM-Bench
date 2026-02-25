#!/usr/bin/env python3
"""
MAS v2 Evaluation -- Train / Test split.

Models:
  HEAD            = Qwen3-VL-4B       (VLM, image+text -> category)
  5 SPECIALISTS   = Qwen3/Sa2VA/LLaVA4D/SpatialRGPT/SpatialReasoner
  FINAL REASONING = DeepSeek-R1       (text-only, SharedMemory + query -> answer)

Usage:
    python run_eval_mas_v2.py \
        --benchmark 3dsrbench \
        --train_ratio 0.5 \
        --seed 42

Or import from Jupyter:
    from run_eval_mas_v2 import build_runners, run_experiment
"""
import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import (
    ALL_CATEGORIES, SPECIALIST_LLMS, ROLES,
    ScoreMap, ScoreMapUpdater,
    run_train, run_test, compute_accuracy,
)
from src2.benchmarks.loaders import load_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



# ======================================================================
# Model loading helpers
# ======================================================================
def build_runners(
    reasoning_api_base: str = "http://localhost:8000/v1",
    reasoning_api_key: str = "EMPTY",
    reasoning_model_name: str = "deepseek-r1",
    specialist_device: str = "cuda",
    use_local_reasoning: bool = False,
    reasoning_local_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
):
    """Instantiate all model runners.

    Returns (head_generate, specialist_generate, reasoning_generate).

    Signatures:
      head_generate(image, prompt) -> str          Qwen3-VL-4B
      specialist_generate(llm_name, image, prompt) -> str
      reasoning_generate(prompt) -> str             DeepSeek-R1 (text-only)

    use_local_reasoning: If True, load DeepSeek-R1-Distill locally (no API).
        Good for H100/single-GPU Jupyter. Uses reasoning_local_model_id.
    """
    from src2.models.qwen3 import Qwen3Runner
    from src2.models.llava import LLaVARunner
    from src2.models.sa2va import Sa2VARunner
    from src2.models.deepseek_r1 import DeepSeekR1Runner, DeepSeekR1LocalRunner

    # --- Head Agent (Qwen3-VL-4B, VLM) ---
    # Reused from specialist cache; loaded once, shared.
    _head_runner = None

    def _get_head():
        nonlocal _head_runner
        if _head_runner is None:
            _head_runner = Qwen3Runner(device=specialist_device)
        return _head_runner

    def head_generate(image, prompt: str) -> str:
        return _get_head().generate(image, prompt, temperature=0.0, max_new_tokens=64)

    # --- 5 Specialist VLMs (lazy-loaded, cached) ---
    _specialist_cache = {}

    def _get_specialist(name: str):
        if name not in _specialist_cache:
            if name == "qwen3_4b":
                _specialist_cache[name] = _get_head()
            elif name == "sa2va":
                _specialist_cache[name] = Sa2VARunner(device=specialist_device)
            elif name == "llava4d":
                _specialist_cache[name] = LLaVARunner(
                    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                    device=specialist_device,
                )
            elif name == "spatial_rgpt":
                from src2.models.spatial_rgpt import SpatialRGPTRunner
                _specialist_cache[name] = SpatialRGPTRunner(device=specialist_device)
            elif name == "spatial_reasoner":
                from src2.models.spatial_reasoner import SpatialReasonerRunner
                _specialist_cache[name] = SpatialReasonerRunner(
                    model_id="ccvl/SpatialReasoner",
                    device=specialist_device,
                )
            else:
                raise ValueError(f"Unknown specialist: {name}")
        return _specialist_cache[name]

    def specialist_generate(llm_name: str, image, prompt: str) -> str:
        runner = _get_specialist(llm_name)
        return runner.generate(image, prompt, temperature=0.0, max_new_tokens=1024)

    # --- Final Reasoning Agent (DeepSeek-R1, text-only) ---
    if use_local_reasoning:
        reasoning = DeepSeekR1LocalRunner(
            model_id=reasoning_local_model_id,
            device=specialist_device,
        )
    else:
        reasoning = DeepSeekR1Runner(
            api_base=reasoning_api_base,
            api_key=reasoning_api_key,
            model_name=reasoning_model_name,
        )

    def reasoning_generate(prompt: str) -> str:
        return reasoning.generate(prompt, temperature=0.0, max_tokens=1024)

    return head_generate, specialist_generate, reasoning_generate


# ======================================================================
# Dataset splitting
# ======================================================================
def split_dataset(dataset, train_ratio: float = 0.5, seed: int = 42):
    """Randomly split a HF dataset into train and test subsets."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_idx = sorted(indices[:split_point])
    test_idx = sorted(indices[split_point:])
    return dataset.select(train_idx), dataset.select(test_idx)


# ======================================================================
# Main experiment runner
# ======================================================================
def run_experiment(
    benchmark: str,
    head_generate,
    specialist_generate,
    reasoning_generate,
    train_ratio: float = 0.5,
    seed: int = 42,
    output_dir: str = None,
    updater: ScoreMapUpdater = None,
    max_samples: int = None,
):
    """Run full MAS v2 experiment: load data -> split -> train -> test -> report."""

    logger.info("Benchmark: %s | Categories: %d (fixed) | Seed: %d", benchmark, len(ALL_CATEGORIES), seed)

    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    logger.info("Loaded %d samples", len(dataset))

    train_ds, test_ds = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
    logger.info("Train: %d | Test: %d", len(train_ds), len(test_ds))

    score_map = ScoreMap(categories=ALL_CATEGORIES, seed=seed)
    updater = updater or ScoreMapUpdater()

    # --- Train phase ---
    logger.info("=" * 60)
    logger.info("TRAIN PHASE")
    logger.info("=" * 60)
    train_results = run_train(
        dataset=train_ds,
        benchmark=benchmark,
        score_map=score_map,
        head_generate=head_generate,
        specialist_generate=specialist_generate,
        reasoning_generate=reasoning_generate,
        updater=updater,
        seed=seed,
    )
    train_metrics = compute_accuracy(train_results)
    logger.info(
        "Train accuracy: %.2f%% (%d/%d)",
        100 * train_metrics["accuracy"],
        train_metrics["correct"], train_metrics["total"],
    )

    # --- Test phase (frozen score map) ---
    logger.info("=" * 60)
    logger.info("TEST PHASE (score map frozen)")
    logger.info("=" * 60)
    test_results = run_test(
        dataset=test_ds,
        benchmark=benchmark,
        score_map=score_map,
        head_generate=head_generate,
        specialist_generate=specialist_generate,
        reasoning_generate=reasoning_generate,
    )
    test_metrics = compute_accuracy(test_results)
    logger.info(
        "Test accuracy: %.2f%% (%d/%d)",
        100 * test_metrics["accuracy"],
        test_metrics["correct"], test_metrics["total"],
    )

    # --- Save results ---
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / ts
        out_path.mkdir(parents=True, exist_ok=True)

        score_map.save(str(out_path / "score_map_final.json"))

        summary = {
            "benchmark": benchmark,
            "seed": seed,
            "train_ratio": train_ratio,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "train_accuracy": train_metrics["accuracy"],
            "train_per_category": train_metrics["per_category"],
            "test_accuracy": test_metrics["accuracy"],
            "test_per_category": test_metrics["per_category"],
            "specialist_llms": SPECIALIST_LLMS,
            "roles": ROLES,
            "timestamp": ts,
        }
        (out_path / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

        with open(out_path / "train_details.jsonl", "w") as f:
            for r in train_results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        with open(out_path / "test_details.jsonl", "w") as f:
            for r in test_results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        logger.info("Results saved to %s", out_path)

    return {
        "score_map": score_map,
        "train_results": train_results,
        "test_results": test_results,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="MAS v2 evaluation")
    parser.add_argument("--benchmark", choices=["3dsrbench", "cvbench"], required=True)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/mas_v2")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--reasoning_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--reasoning_api_key", type=str, default="EMPTY")
    parser.add_argument("--reasoning_model_name", type=str, default="deepseek-r1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    head_gen, spec_gen, reason_gen = build_runners(
        reasoning_api_base=args.reasoning_api_base,
        reasoning_api_key=args.reasoning_api_key,
        reasoning_model_name=args.reasoning_model_name,
        specialist_device=args.device,
    )

    out_dir = f"{args.output_dir}/{args.benchmark}"
    run_experiment(
        benchmark=args.benchmark,
        head_generate=head_gen,
        specialist_generate=spec_gen,
        reasoning_generate=reason_gen,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=out_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
