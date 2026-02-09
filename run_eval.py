#!/usr/bin/env python3
"""
STVQA-7K evaluation: Qwen2.5-VL, LLaVA, GPT, Gemini.
Usage:
  python run_eval.py --models qwen llava gemini --split val --max_samples 50
  GEMINI_API_KEY=... python run_eval.py --models gemini
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm

from src.data import (
    load_stvqa,
    get_prompt,
    normalize_answer_only,
    accuracy,
)
from src.models.qwen import QwenRunner
from src.models.llava import LLaVARunner
from src.models.gpt import GPTRunner
from src.models.gemini import GeminiRunner


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_model(model_name: str, dataset, config: dict, output_dir: Path):
    eval_cfg = config.get("eval", {})
    temp = eval_cfg.get("temperature", 0.0)
    max_new = eval_cfg.get("max_new_tokens", 512)
    prompts = [get_prompt(dataset[i]) for i in range(len(dataset))]
    # STVQA-7K uses "images" column (PIL)
    images = [dataset[i].get("images") or dataset[i].get("image") for i in range(len(dataset))]
    gt = [dataset[i]["answer_only"] for i in range(len(dataset))]

    if model_name == "qwen":
        m_cfg = config.get("models", {}).get("qwen", {})
        if not m_cfg.get("enabled", True):
            return None
        runner = QwenRunner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "llava":
        m_cfg = config.get("models", {}).get("llava", {})
        if not m_cfg.get("enabled", True):
            return None
        runner = LLaVARunner(
            model_id=m_cfg.get("model_id", "llava-hf/llava-1.5-7b-hf"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "gpt":
        m_cfg = config.get("models", {}).get("gpt", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            print(f"[skip] {model_name}: no OPENAI_API_KEY")
            return None
        runner = GPTRunner(
            model_id=m_cfg.get("model_id", "gpt-4o"),
            api_key=api_key,
        )
    elif model_name == "gemini":
        m_cfg = config.get("models", {}).get("gemini", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(f"[skip] {model_name}: no GEMINI_API_KEY / GOOGLE_API_KEY")
            return None
        runner = GeminiRunner(
            model_id=m_cfg.get("model_id", "gemini-2.0-flash"),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    preds = []
    for i in tqdm(range(len(dataset)), desc=model_name):
        img = images[i]
        prompt = prompts[i]
        if model_name in ("gpt", "gemini"):
            out = runner.generate(img, prompt, temperature=temp, max_tokens=max_new)
        else:
            out = runner.generate(img, prompt, temperature=temp, max_new_tokens=max_new)
        letter = normalize_answer_only(out)
        preds.append(letter)

    acc = accuracy(preds, gt)
    results = {
        "model": model_name,
        "accuracy": acc,
        "num_samples": len(gt),
        "predictions": preds if config.get("output", {}).get("save_predictions") else None,
        "ground_truth": gt if config.get("output", {}).get("save_predictions") else None,
    }

    if config.get("output", {}).get("per_category_accuracy") and "category" in dataset.features:
        by_cat = {}
        for i in range(len(dataset)):
            c = dataset[i]["category"]
            if c not in by_cat:
                by_cat[c] = {"pred": [], "gt": []}
            by_cat[c]["pred"].append(preds[i])
            by_cat[c]["gt"].append(gt[i])
        results["by_category"] = {
            c: accuracy(d["pred"], d["gt"]) for c, d in by_cat.items()
        }

    out_path = output_dir / f"{model_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                k: v
                for k, v in results.items()
                if k not in ("predictions", "ground_truth")
                or config.get("output", {}).get("save_predictions")
            },
            f,
            indent=2,
        )
    if config.get("output", {}).get("save_predictions"):
        pred_path = output_dir / f"{model_name}_preds.jsonl"
        with open(pred_path, "w") as f:
            for i in range(len(dataset)):
                correct = preds[i] == gt[i]
                rec = {
                    "idx": i,
                    "pred": preds[i],
                    "gt": gt[i],
                    "correct": correct,
                    "category": dataset[i].get("category"),
                    "question_only": dataset[i].get("question_only"),
                    "options": dataset[i].get("options"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="STVQA-7K evaluation")
    parser.add_argument("--config", default="config.yaml", help="Config YAML")
    parser.add_argument("--models", nargs="+", default=["qwen", "llava", "gemini"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    if args.max_samples is not None:
        ds_cfg = {**ds_cfg, "max_samples": args.max_samples}

    dataset = load_stvqa(
        dataset_name=ds_cfg.get("name", "OX-PIXL/STVQA-7K"),
        split=args.split,
        max_samples=ds_cfg.get("max_samples"),
    )
    print(f"Loaded {len(dataset)} samples (split={args.split})")

    output_dir = Path(args.output_dir or config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f)

    all_results = []
    for model_name in args.models:
        try:
            res = run_model(model_name, dataset, config, run_dir)
            if res is not None:
                all_results.append(res)
                print(f"{model_name}: accuracy = {res['accuracy']:.4f} ({res['num_samples']} samples)")
        except Exception as e:
            print(f"{model_name}: error - {e}")
            raise

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            [{"model": r["model"], "accuracy": r["accuracy"], "num_samples": r["num_samples"]}
            for r in all_results
        ],
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
