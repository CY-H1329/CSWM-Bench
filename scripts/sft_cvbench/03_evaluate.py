#!/usr/bin/env python3
"""
SFT Evaluation script for CV-Bench.

Evaluates a trained model on human_selected_test_set.
Computes: 2D accuracy, 3D accuracy, overall accuracy.

Usage:
  python scripts/sft_cvbench/03_evaluate.py --model qwen3_4b --shots 10 --checkpoint results/sft_cvbench/checkpoints/qwen3_4b_cvbench_10shot
"""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from tqdm import tqdm

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
)
from src.data import normalize_answer_only, accuracy

# 2D = Count, Relation | 3D = Depth, Distance
CVBENCH_2D = {"Count", "Relation"}
CVBENCH_3D = {"Depth", "Distance"}


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _eval_qwen3_baseline(indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate Qwen3-VL base model (zero-shot)."""
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
    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="qwen3_4b"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
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
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_qwen3(checkpoint: Path, indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate Qwen3-VL SFT checkpoint (LoRA)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    base_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="qwen3_4b"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
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
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_llava_baseline(indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate LLaVA-1.5 base model (zero-shot)."""
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
    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    for idx in tqdm(indices, desc="llava4d"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
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
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_llava(checkpoint: Path, indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate LLaVA-1.5 SFT checkpoint (LoRA)."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    base_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    for idx in tqdm(indices, desc="llava4d"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
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
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_sa2va_baseline(indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate Sa2VA-4B base model (zero-shot)."""
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
    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="sa2va"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
            continue
        img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        input_dict = {"image": img_rgb, "text": f"<image>{query}", "past_text": "", "mask_prompts": None, "tokenizer": tokenizer}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*generation_config.*")
            out = model.predict_forward(**input_dict)
        text = (out.get("prediction") or "").strip()
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_sa2va(checkpoint: Path, indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate Sa2VA-4B SFT checkpoint (LoRA on language_model)."""
    import warnings
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    from peft import PeftModel

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
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=False,
        )
        lm_adapter = checkpoint / "language_model"
        if lm_adapter.exists():
            model.language_model = PeftModel.from_pretrained(model.language_model, str(lm_adapter))
    finally:
        torch.linspace = _orig_linspace
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "_count"):
        model._count = 0
    model.eval()
    if hasattr(model, "preparing_for_generation"):
        model.preparing_for_generation(tokenizer, max_new_tokens=max_new_tokens)

    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="sa2va"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
            continue
        img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        input_dict = {"image": img_rgb, "text": f"<image>{query}", "past_text": "", "mask_prompts": None, "tokenizer": tokenizer}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*generation_config.*")
            out = model.predict_forward(**input_dict)
        text = (out.get("prediction") or "").strip()
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_spatialreasoner_baseline(indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate SpatialReasoner base model (zero-shot)."""
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
    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    for idx in tqdm(indices, desc="spatialreasoner"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
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
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _eval_spatialreasoner(checkpoint: Path, indices: list, max_new_tokens: int = 256) -> tuple:
    """Evaluate SpatialReasoner SFT checkpoint (LoRA)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    base_id = "ccvl/SpatialReasoner"
    processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="spatialreasoner"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        in_len = inputs["input_ids"].shape[1]
        text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)
        categories.append(cat)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def _compute_results(preds, gt_list, categories):
    """Compute accuracy by 2D/3D and per task."""
    acc_overall = accuracy(preds, gt_list)
    by_cat = {}
    for p, g, c in zip(preds, gt_list, categories):
        c = str(c).strip() if c else "unknown"
        if c not in by_cat:
            by_cat[c] = {"preds": [], "gts": []}
        by_cat[c]["preds"].append(p)
        by_cat[c]["gts"].append(g)
    task_acc = {k: accuracy(v["preds"], v["gts"]) for k, v in by_cat.items()}
    preds_2d = [p for p, c in zip(preds, categories) if str(c).strip() in CVBENCH_2D]
    gt_2d = [g for g, c in zip(gt_list, categories) if str(c).strip() in CVBENCH_2D]
    preds_3d = [p for p, c in zip(preds, categories) if str(c).strip() in CVBENCH_3D]
    gt_3d = [g for g, c in zip(gt_list, categories) if str(c).strip() in CVBENCH_3D]
    acc_2d = accuracy(preds_2d, gt_2d) if preds_2d else 0.0
    acc_3d = accuracy(preds_3d, gt_3d) if preds_3d else 0.0
    task_acc_full = {"Count": 0.0, "Relation": 0.0, "Depth": 0.0, "Distance": 0.0}
    for k, v in task_acc.items():
        if k in task_acc_full:
            task_acc_full[k] = v
    return acc_overall, acc_2d, acc_3d, task_acc_full


def _eval_spatialrgpt_baseline(indices: list, model_id: str = "a8cheng/SpatialRGPT-VILA1.5-8B", max_new_tokens: int = 256) -> tuple:
    """Evaluate SpatialRGPT base model (zero-shot, no training). Requires SPATIALRGPT_PATH."""
    import os
    if not os.environ.get("SPATIALRGPT_PATH") or not Path(os.environ["SPATIALRGPT_PATH"]).is_dir():
        raise RuntimeError(
            "SPATIALRGPT_PATH not set. Clone SpatialRGPT and set: export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
        )
    from src2.models.spatial_rgpt import SpatialRGPTRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = SpatialRGPTRunner(model_id=model_id, device=device)
    ds = load_benchmark("cvbench", max_samples=None, seed=42)
    preds, gt_list, categories = [], [], []
    for idx in tqdm(indices, desc="spatialrgpt"):
        ex = ds[idx]
        img = get_benchmark_image(ex, "cvbench")
        query = get_benchmark_prompt(ex, "cvbench")
        gt = get_benchmark_answer(ex, "cvbench")
        cat = get_benchmark_category(ex, "cvbench") or "unknown"
        if img is None:
            preds.append("")
            gt_list.append(gt)
            categories.append(cat)
            continue
        text = runner.generate(img, query, temperature=0.0, max_new_tokens=max_new_tokens)
        preds.append(normalize_answer_only(text))
        gt_list.append(gt)
        categories.append(cat)
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, gt_list, categories


def parse_args():
    parser = argparse.ArgumentParser(description="SFT evaluation on CV-Bench")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen3_4b", "llava4d", "sa2va", "spatialrgpt", "spatialreasoner"],
    )
    parser.add_argument("--shots", type=int, required=True, choices=[0, 10, 30, 100], help="0 = baseline (no training)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path; use 'base' for spatialrgpt baseline")
    parser.add_argument("--split", type=str, default="human_test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit for debug")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config_sft.yaml"
    config = load_config(config_path)

    splits_dir = Path(config["paths"]["splits_dir"])
    split_path = splits_dir / f"{args.split}.json"
    if not split_path.exists():
        print(f"ERROR: Missing split: {split_path}. Run 01_sample_dataset.py first.")
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)
    indices = split_data["indices"]
    if args.max_samples:
        indices = indices[: args.max_samples]

    out_dir = args.output_dir or Path(config["paths"]["output_dir"]) / args.model / str(args.shots) / args.split
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SFT Evaluation (CV-Bench)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Shots: {args.shots}" + (" (baseline, no training)" if args.shots == 0 else ""))
    print(f"Checkpoint: {args.checkpoint or 'base'}")
    print(f"Split: {args.split} ({len(indices)} samples)")
    print(f"Output: {out_dir}")
    print()

    use_baseline = args.shots == 0 or args.checkpoint == "base"
    if not use_baseline:
        if not args.checkpoint:
            print("ERROR: --checkpoint required (use --shots 0 for baseline)")
            sys.exit(1)
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"ERROR: Checkpoint not found: {ckpt}")
            sys.exit(1)
    else:
        ckpt = None

    preds, gt_list, categories = None, None, None
    if use_baseline:
        if args.model == "qwen3_4b":
            preds, gt_list, categories = _eval_qwen3_baseline(indices, args.max_new_tokens)
        elif args.model == "llava4d":
            preds, gt_list, categories = _eval_llava_baseline(indices, args.max_new_tokens)
        elif args.model == "sa2va":
            preds, gt_list, categories = _eval_sa2va_baseline(indices, args.max_new_tokens)
        elif args.model == "spatialrgpt":
            preds, gt_list, categories = _eval_spatialrgpt_baseline(indices, max_new_tokens=args.max_new_tokens)
        elif args.model == "spatialreasoner":
            preds, gt_list, categories = _eval_spatialreasoner_baseline(indices, args.max_new_tokens)
        else:
            print(f"Model {args.model}: no baseline eval.")
            sys.exit(1)
    elif args.model == "qwen3_4b":
        preds, gt_list, categories = _eval_qwen3(ckpt, indices, args.max_new_tokens)
    elif args.model == "llava4d":
        preds, gt_list, categories = _eval_llava(ckpt, indices, args.max_new_tokens)
    elif args.model == "sa2va":
        preds, gt_list, categories = _eval_sa2va(ckpt, indices, args.max_new_tokens)
    elif args.model == "spatialreasoner":
        preds, gt_list, categories = _eval_spatialreasoner(ckpt, indices, args.max_new_tokens)
    else:
        print(f"Model {args.model}: no SFT eval (use base model eval or LLaMA-Factory). Saving placeholder.")
        results = {
            "model": args.model,
            "shots": args.shots,
            "split": args.split,
            "n_samples": len(indices),
            "overall_accuracy": 0.0,
            "accuracy_2d": 0.0,
            "accuracy_3d": 0.0,
            "task_accuracy": {"Count": 0.0, "Relation": 0.0, "Depth": 0.0, "Distance": 0.0},
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Placeholder saved to {out_dir / 'results.json'}")
        return

    acc_overall, acc_2d, acc_3d, task_acc = _compute_results(preds, gt_list, categories)
    results = {
        "model": args.model,
        "shots": args.shots,
        "split": args.split,
        "n_samples": len(indices),
        "overall_accuracy": acc_overall,
        "accuracy_2d": acc_2d,
        "accuracy_3d": acc_3d,
        "task_accuracy": task_acc,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"Overall: {acc_overall:.4f} | 2D: {acc_2d:.4f} | 3D: {acc_3d:.4f}")
    print(f"Task: {task_acc}")
    print("=" * 70)
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
