#!/usr/bin/env python3
"""
STVQA-7K Evaluation: hunarbatra/STVQA-7K val split.

Evaluates 5 models: Qwen-3.0-VL 4B, LLaVA-4D, SpatialReasoner, SpatialRGPT, Sa2VA.

Usage:
  python scripts/stvqa7k/eval_stvqa7k.py --model qwen3_4b [--max_samples 10]
  python scripts/stvqa7k/eval_stvqa7k.py --model all  # run all 5 models
"""
import argparse
import gc
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.data import load_stvqa, get_prompt, normalize_answer_only, accuracy


def get_stvqa_image(example):
    """Extract PIL Image from hunarbatra/STVQA-7K example."""
    img = example.get("images")
    if img is None:
        return None
    if hasattr(img, "convert"):
        return img.convert("RGB") if img.mode != "RGB" else img
    if isinstance(img, dict):
        sub = img.get("image")
        if sub is not None and hasattr(sub, "convert"):
            return sub.convert("RGB") if sub.mode != "RGB" else sub
        raw = img.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def _eval_qwen3_4b(dataset, indices, max_new_tokens=256):
    """Qwen-3.0-VL 4B (Qwen/Qwen3-VL-4B-Instruct)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    base_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds, gt_list = [], []
    for idx in tqdm(indices, desc="qwen3_4b"):
        ex = dataset[idx]
        img = get_stvqa_image(ex)
        query = get_prompt(ex, include_options=True)
        gt = (ex.get("answer_only") or "").strip().upper()
        if gt and gt not in "ABCD":
            gt = gt[0] if gt else ""
        if img is None:
            preds.append("")
            gt_list.append(gt)
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list


def _eval_llava4d(dataset, indices, max_new_tokens=256):
    """LLaVA-4D (llava-hf/llava-1.5-7b-hf)."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    base_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    preds, gt_list = [], []
    for idx in tqdm(indices, desc="llava4d"):
        ex = dataset[idx]
        img = get_stvqa_image(ex)
        query = get_prompt(ex, include_options=True)
        gt = (ex.get("answer_only") or "").strip().upper()
        if gt and gt not in "ABCD":
            gt = gt[0] if gt else ""
        if img is None:
            preds.append("")
            gt_list.append(gt)
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]}]
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
        except Exception:
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=[img.convert("RGB")], text=[prompt], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        in_len = inputs["input_ids"].shape[1]
        text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list


def _eval_sa2va(dataset, indices, max_new_tokens=256):
    """Sa2VA-4B (ByteDance/Sa2VA-4B)."""
    import warnings
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    def _patch_sa2va():
        if hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
            _orig = PreTrainedModel.mark_tied_weights_as_initialized
            def _patched(self):
                if not hasattr(self, "all_tied_weights_keys"):
                    old = getattr(self, "_tied_weights_keys", None)
                    if old is not None and hasattr(old, "keys"):
                        self.all_tied_weights_keys = old
                    elif isinstance(old, (list, tuple)):
                        keys = []
                        for x in old:
                            keys.extend(x if isinstance(x, (list, tuple)) else [x])
                        self.all_tied_weights_keys = {k: None for k in keys}
                    else:
                        self.all_tied_weights_keys = {}
                _orig(self)
            PreTrainedModel.mark_tied_weights_as_initialized = _patched

    _patch_sa2va()
    _orig_linspace = torch.linspace
    def _patched_linspace(*a, **kw):
        kw.setdefault("device", torch.device("cpu"))
        return _orig_linspace(*a, **kw)

    try:
        torch.linspace = _patched_linspace
        base_id = "ByteDance/Sa2VA-4B"
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=False,
        )
    finally:
        torch.linspace = _orig_linspace

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "_count"):
        model._count = 0
    model.eval()
    if hasattr(model, "preparing_for_generation"):
        model.preparing_for_generation(tokenizer, max_new_tokens=max_new_tokens)

    preds, gt_list = [], []
    for idx in tqdm(indices, desc="sa2va"):
        ex = dataset[idx]
        img = get_stvqa_image(ex)
        query = get_prompt(ex, include_options=True)
        gt = (ex.get("answer_only") or "").strip().upper()
        if gt and gt not in "ABCD":
            gt = gt[0] if gt else ""
        if img is None:
            preds.append("")
            gt_list.append(gt)
            continue
        img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        input_dict = {"image": img_rgb, "text": f"<image>{query}", "past_text": "", "mask_prompts": None, "tokenizer": tokenizer}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*")
            out = model.predict_forward(**input_dict)
        text = (out.get("prediction") or "").strip()
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list


def _eval_spatialreasoner(dataset, indices, max_new_tokens=512):
    """SpatialReasoner (ccvl/SpatialReasoner)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base_id = "ccvl/SpatialReasoner"
    processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    preds, gt_list = [], []
    for idx in tqdm(indices, desc="spatialreasoner"):
        ex = dataset[idx]
        img = get_stvqa_image(ex)
        query = get_prompt(ex, include_options=True)
        gt = (ex.get("answer_only") or "").strip().upper()
        if gt and gt not in "ABCD":
            gt = gt[0] if gt else ""
        if img is None:
            preds.append("")
            gt_list.append(gt)
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        in_len = inputs["input_ids"].shape[1]
        text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list


def _eval_spatialrgpt(dataset, indices, max_new_tokens=256):
    """SpatialRGPT (a8cheng/SpatialRGPT-VILA1.5-8B). Requires SPATIALRGPT_PATH."""
    import os
    import subprocess
    if not os.environ.get("SPATIALRGPT_PATH") or not Path(os.environ["SPATIALRGPT_PATH"]).is_dir():
        raise RuntimeError(
            "SPATIALRGPT_PATH not set. Clone SpatialRGPT and set: export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
        )
    # Patch vision_encoder.py for Python 3.9 (match requires 3.10+) before import
    patch_script = ROOT / "scripts" / "stvqa7k" / "patch_spatialrgpt_py39.py"
    if patch_script.exists():
        subprocess.run([sys.executable, str(patch_script)], check=False, capture_output=True)
    from src2.models.spatial_rgpt import SpatialRGPTRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = SpatialRGPTRunner(model_id="a8cheng/SpatialRGPT-VILA1.5-8B", device=device)

    preds, gt_list = [], []
    for idx in tqdm(indices, desc="spatialrgpt"):
        ex = dataset[idx]
        img = get_stvqa_image(ex)
        query = get_prompt(ex, include_options=True)
        gt = (ex.get("answer_only") or "").strip().upper()
        if gt and gt not in "ABCD":
            gt = gt[0] if gt else ""
        if img is None:
            preds.append("")
            gt_list.append(gt)
            continue
        text = runner.generate(img, query, temperature=0.0, max_new_tokens=max_new_tokens)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list


EVAL_FNS = {
    "qwen3_4b": _eval_qwen3_4b,
    "llava4d": _eval_llava4d,
    "sa2va": _eval_sa2va,
    "spatialreasoner": _eval_spatialreasoner,
    "spatialrgpt": _eval_spatialrgpt,
}


def parse_args():
    p = argparse.ArgumentParser(description="STVQA-7K evaluation (hunarbatra/STVQA-7K val)")
    p.add_argument("--model", type=str, required=True, choices=list(EVAL_FNS) + ["all"])
    p.add_argument("--dataset", type=str, default="hunarbatra/STVQA-7K")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "stvqa7k"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading STVQA-7K...")
    dataset = load_stvqa(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )
    indices = list(range(len(dataset)))
    print(f"  {len(indices)} samples")

    models = list(EVAL_FNS) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models:
        print("\n" + "=" * 60)
        print(f"Evaluating: {model_name}")
        print("=" * 60)
        fn = EVAL_FNS[model_name]
        preds, gt_list = fn(dataset, indices, max_new_tokens=args.max_new_tokens)
        acc = accuracy(preds, gt_list)
        all_results[model_name] = {"accuracy": acc, "n_samples": len(indices)}
        print(f"  Accuracy: {acc:.4f}")

        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        with open(model_out / "results.json", "w") as f:
            json.dump({"model": model_name, "accuracy": acc, "n_samples": len(indices)}, f, indent=2)
        with open(model_out / "predictions.jsonl", "w") as f:
            for i, (p, g) in enumerate(zip(preds, gt_list)):
                f.write(json.dumps({"idx": i, "pred": p, "gt": g}) + "\n")

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for m, r in all_results.items():
        print(f"  {m}: {r['accuracy']:.4f}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
