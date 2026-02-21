#!/usr/bin/env python3
"""
SpatialLLM / SpatialReasoner evaluation on 3DSRBench.

Uses ccvl/SpatialReasoner (Qwen2.5-VL based, SOTA on 3DSRBench).
Default: 100 samples. Run on H100.

Usage:
  python evals_spatialllm/run_spatialllm_3dsrbench.py
  python evals_spatialllm/run_spatialllm_3dsrbench.py --max_samples 100 --seed 42
  python evals_spatialllm/run_spatialllm_3dsrbench.py --model_id ccvl/SpatialReasoner-SFT
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
)
from src.data import normalize_answer_only


def _canonical_letter(s: str) -> str:
    """Extract A/B/C/D for comparison."""
    if not s:
        return ""
    s = str(s).strip().upper()
    if s in "ABCD":
        return s
    m = re.search(r"\(([A-D])\)", s)
    return m.group(1) if m else ""


def _build_prompt(question: str) -> str:
    """Simple VQA prompt for 3DSRBench."""
    return f"""Answer the following spatial reasoning question based on the image.

Question: {question}

Provide your final answer as (A), (B), (C), or (D)."""


def load_spatial_reasoner(model_id: str, device: str = "cuda"):
    """Load SpatialReasoner (Qwen2.5-VL based) — direct model+processor."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError("SpatialReasoner requires transformers>=4.50. pip install transformers>=4.50")

    import torch

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Processor: ccvl/SpatialReasoner* has compat issues with AutoProcessor; use base Qwen2.5-VL
    if "ccvl/SpatialReasoner" in model_id:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="SpatialLLM/SpatialReasoner on 3DSRBench")
    parser.add_argument("--model_id", default="ccvl/SpatialReasoner", help="HuggingFace model (ccvl/SpatialReasoner, ccvl/SpatialReasoner-SFT, ccvl/SpatialReasoner-Zero)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/evals_spatialllm")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading 3DSRBench (max_samples={args.max_samples})...")
    ds = load_benchmark("3dsrbench", max_samples=args.max_samples, seed=args.seed)
    print(f"  {len(ds)} samples")

    print(f"Loading {args.model_id}...")
    model, processor = load_spatial_reasoner(args.model_id, args.device)

    import torch
    out_dir = Path(args.output_dir)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(len(ds)), desc="SpatialReasoner"):
        ex = ds[i]
        img = get_benchmark_image(ex, "3dsrbench")
        query = get_benchmark_prompt(ex, "3dsrbench")
        gt = get_benchmark_answer(ex, "3dsrbench")

        if img is None:
            results.append({"idx": i, "error": "no_image", "gt": gt})
            continue

        prompt = _build_prompt(query)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = normalize_answer_only(response)
            is_correct = _canonical_letter(pred) == _canonical_letter(gt)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "idx": i,
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "response": response[:500],
            })
        except Exception as e:
            results.append({"idx": i, "error": str(e), "gt": gt})

    acc = correct / total if total else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.2%}")

    summary = {
        "model_id": args.model_id,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
