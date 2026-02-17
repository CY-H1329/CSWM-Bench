"""
Unified loaders for all 4 benchmarks.
Returns normalized format: image, question, options (list or None), answer, category (optional).
"""
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
import io
import requests

# Task categories for Head-Agent classification (do NOT give to Head - it must infer)
SPATIAL_TASK_CATEGORIES = [
    "depth",
    "distance",
    "relation",
    "existence",
    "count",
    "instance_location",
    "orientation",
    "size",
    "reach",
]

# GQA: lmms-lab has images and instructions in separate configs; we merge them
GQA_STRUCTURAL_CATEGORIES = ["query", "verify", "choose", "logical", "compare"]
GQA_SEMANTIC_CATEGORIES = ["relation", "attribute", "object", "global", "other"]

BENCHMARK_CONFIGS = {
    "gqa": {
        "name": "lmms-lab/GQA",
        "instructions_config": "val_balanced_instructions",
        "images_config": "val_balanced_images",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answer",
        "category_key": "semantic",  # or "structural" from groups.types
    },
    "cvbench": {
        "name": "nyu-visionx/CV-Bench",
        "split": "test",
        "image_key": "image",
        "question_key": "question",
        "options_key": "choices",
        "answer_key": "answer",
        "category_key": "task",
    },
    "3dsrbench": {
        "name": "ccvl/3DSRBench",
        "split": "test",
        "subset": "benchmark",
        "image_key": "image_url",
        "question_key": "question",
        "options_keys": ["A", "B", "C", "D"],
        "answer_key": "answer",
        "category_key": "category",
    },
}


def _fetch_image_from_url(url: str) -> Optional[Image.Image]:
    """Fetch image from URL. Returns PIL Image or None."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _load_gqa(max_samples: Optional[int] = None, max_per_category: Optional[int] = None, seed: int = 42):
    """Load GQA: merge val_balanced_instructions with val_balanced_images."""
    cfg = BENCHMARK_CONFIGS["gqa"]
    name = cfg["name"]
    inst_cfg = cfg["instructions_config"]
    img_cfg = cfg["images_config"]

    instructions = load_dataset(name, inst_cfg, split="val", trust_remote_code=True)
    images_ds = load_dataset(name, img_cfg, split="val", trust_remote_code=True)
    # imageId in instructions may be int or str
    img_by_id = {}
    for ex in images_ds:
        k = str(ex.get("id", ""))
        img_by_id[k] = ex.get("image")

    merged = []
    for i, ex in enumerate(instructions):
        img_id = str(ex.get("imageId") or ex.get("image_id") or "")
        img = img_by_id.get(img_id)
        if img is None:
            continue
        groups = ex.get("groups") or {}
        types = groups.get("types") or {}
        semantic = types.get("semantic") or "unknown"
        structural = types.get("structural") or "unknown"
        merged.append({
            "idx": i,
            "image": img.convert("RGB") if hasattr(img, "convert") else img,
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "semantic": semantic,
            "structural": structural,
            "category": semantic,
        })

    from datasets import Dataset
    ds = Dataset.from_list(merged)

    rng = random.Random(seed)
    if max_per_category is not None:
        by_cat = {}
        for i in range(len(ds)):
            c = ds[i].get("category") or "unknown"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            k = min(max_per_category, len(by_cat[c]))
            indices.extend(rng.sample(by_cat[c], k))
        indices.sort()
        ds = ds.select(indices)
    elif max_samples is not None:
        n = min(max_samples, len(ds))
        indices = rng.sample(range(len(ds)), n)
        indices.sort()
        ds = ds.select(indices)

    return ds


def load_benchmark(
    benchmark: str,
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    seed: int = 42,
):
    """
    Load a benchmark dataset. Uses HuggingFace cache (from setup_datasets.py).
    Returns dataset with normalized access via get_benchmark_* helpers.
    """
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

    if benchmark == "gqa":
        return _load_gqa(max_samples=max_samples, max_per_category=max_per_category, seed=seed)

    cfg = BENCHMARK_CONFIGS[benchmark]
    name = cfg["name"]
    split = cfg["split"]
    subset = cfg.get("subset")

    if subset:
        ds = load_dataset(name, subset, split=split)
    else:
        ds = load_dataset(name, split=split)

    rng = random.Random(seed)
    cat_key = cfg.get("category_key")

    if max_per_category is not None and cat_key and cat_key in ds.features:
        by_cat = {}
        for i in range(len(ds)):
            c = ds[i].get(cat_key) or "unknown"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            idx_list = by_cat[c]
            k = min(max_per_category, len(idx_list))
            indices.extend(rng.sample(idx_list, k))
        indices.sort()
        ds = ds.select(indices)
    elif max_samples is not None:
        n = min(max_samples, len(ds))
        indices = rng.sample(range(len(ds)), n)
        indices.sort()
        ds = ds.select(indices)

    return ds


def get_benchmark_image(example: Dict, benchmark: str) -> Optional[Image.Image]:
    """Extract PIL Image from example."""
    if benchmark == "gqa":
        img = example.get("image")
        if img is not None and hasattr(img, "convert"):
            return img.convert("RGB")
        return img

    cfg = BENCHMARK_CONFIGS[benchmark]
    img_key = cfg["image_key"]

    if img_key == "image_url":
        url = example.get(img_key)
        if url:
            return _fetch_image_from_url(url)
        return None

    img = example.get("images") or example.get("image")
    if img is None:
        return None
    if hasattr(img, "convert"):
        return img.convert("RGB")
    return img


def get_benchmark_prompt(example: Dict, benchmark: str, include_options: bool = True) -> str:
    """Build prompt (question + options if any)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    q_key = cfg["question_key"]
    question = example.get(q_key) or ""

    if not include_options:
        return question

    opts_key = cfg.get("options_key")
    opts_keys = cfg.get("options_keys")

    if opts_key and opts_key in example:
        opts = example[opts_key]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)
    elif opts_keys:
        opts = [example.get(k) for k in opts_keys if example.get(k)]
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)

    return question


def get_benchmark_answer(example: Dict, benchmark: str) -> str:
    """Ground-truth answer (letter A/B/C/D or raw string)."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    ans_key = cfg["answer_key"]
    ans = example.get(ans_key) or ""
    s = str(ans).strip()
    # Benchmarks with options: extract letter (CV-Bench uses "(C)")
    if cfg.get("options_key") or cfg.get("options_keys"):
        for c in "ABCDEF":
            if f"({c})" in s.upper() or s.upper() == c:
                return c
    return s


def get_benchmark_category(example: Dict, benchmark: str) -> Optional[str]:
    """Category if available (for evaluation only - never pass to Head-Agent)."""
    if benchmark == "gqa":
        return example.get("category") or example.get("semantic")
    cfg = BENCHMARK_CONFIGS[benchmark]
    cat_key = cfg.get("category_key")
    if cat_key and cat_key in example:
        return str(example[cat_key])
    return None
