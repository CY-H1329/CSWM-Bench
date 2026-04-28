#!/usr/bin/env python3
"""
Stage 1: VLM baseline on CausalSpatial (MCQ A/B/C/D).

Model: Qwen2.5-VL-7B-Instruct (HF).

Outputs:
  checkpoints/causalspatial_wm/stage1/<model_key>/<split>/details.jsonl
  checkpoints/causalspatial_wm/stage1/<model_key>/<split>/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

from scripts.evals.causalspatial_wm.load_causalspatial import TASK_TYPES, load_all
from scripts.evals.causalspatial_wm.utils_mcq import build_mcq_prompt, extract_letter, accuracy


ROOT = Path(__file__).resolve().parents[3]


def _load_images(paths: List[str]) -> List[Image.Image]:
    ims = []
    for p in paths:
        if not p:
            continue
        ims.append(Image.open(p).convert("RGB"))
    return ims


def _load_qwen25_vl(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, proc


def _generate(model, processor, images: List[Image.Image], prompt: str, device: str, max_new_tokens: int = 32) -> str:
    import torch

    # Qwen-VL chat template is handled by qwen-vl-utils in many setups, but keep it generic:
    # Provide a single user message with images + text.
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # decode only generated part if possible
    decoded = processor.batch_decode(out, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_per_task", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--checkpoints_dir", default=str(ROOT / "checkpoints" / "causalspatial_wm"))
    args = ap.parse_args()

    items = load_all(split=args.split, max_samples_per_task=args.max_per_task, checkpoints_dir=args.checkpoints_dir)

    model, proc = _load_qwen25_vl(args.model_id, args.device)

    run_dir = Path(args.checkpoints_dir) / "stage1" / "qwen2_5_vl_7b" / args.split / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    details_path = run_dir / "details.jsonl"

    preds = []
    gts = []
    by_task = defaultdict(lambda: {"preds": [], "gts": []})
    details = []

    for ex in tqdm(items, desc="stage1"):
        images = _load_images(ex["image_paths"])
        prompt = build_mcq_prompt(ex["question"], ex["choices"])
        try:
            resp = _generate(model, proc, images, prompt, args.device, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            resp = ""
            pred = ""
            err = str(e)
        else:
            pred = extract_letter(resp)
            err = ""

        gt = ex.get("answer", "")
        preds.append(pred)
        gts.append(gt)
        by_task[ex["task_type"]]["preds"].append(pred)
        by_task[ex["task_type"]]["gts"].append(gt)
        d = {
            "sample_id": ex["sample_id"],
            "task_type": ex["task_type"],
            "pred": pred,
            "gt": gt,
            "response": resp,
        }
        if err:
            d["error"] = err
        details.append(d)

    with open(details_path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    results = {
        "stage": "stage1",
        "model_id": args.model_id,
        "split": args.split,
        "total": {"n": len(preds), "accuracy": accuracy(preds, gts)},
        "per_task": {
            t: {"n": len(v["preds"]), "accuracy": accuracy(v["preds"], v["gts"])} for t, v in by_task.items()
        },
        "run_dir": run_dir.as_posix(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()

