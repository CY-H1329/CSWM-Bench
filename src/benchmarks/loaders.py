"""
Unified loaders for all 4 benchmarks.
Returns normalized format: image, question, options (list or None), answer, category (optional).

Supports frozen benchmarks: when use_frozen=True (default), loads from data/frozen_benchmarks/
for reproducible paper experiments. Set use_frozen=False to load from HuggingFace with sampling.
"""
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
import io
import requests

# Frozen benchmark paths (DO NOT MODIFY - used for all paper experiments)
FROZEN_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "frozen_benchmarks"
FROZEN_PATHS = {
    "cvbench": "cvbench_400",
    "3dsrbench": "3dsrbench_500",
}

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

BENCHMARK_CONFIGS = {
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


def load_benchmark(
    benchmark: str,
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    category_filter: Optional[List[str]] = None,
    seed: int = 42,
    use_frozen: bool = True,
):
    """
    Load a benchmark dataset.

    When use_frozen=True (default): loads from data/frozen_benchmarks/ for reproducible
    paper experiments. Ignores max_samples/max_per_category - returns exact frozen set.

    When use_frozen=False: loads from HuggingFace cache and applies sampling.

    Returns dataset with normalized access via get_benchmark_* helpers.

    Args:
        category_filter: If set (and use_frozen=False), keep only samples whose category is in this list.
    """
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

    # Try frozen benchmark first
    if use_frozen and benchmark in FROZEN_PATHS:
        frozen_name = FROZEN_PATHS[benchmark]
        frozen_path = FROZEN_BENCHMARK_DIR / frozen_name
        if frozen_path.exists() and (frozen_path / "dataset_info.json").exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(frozen_path))
            return ds

    # Fallback: load from HuggingFace
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

    if category_filter is not None and cat_key and cat_key in ds.features:
        cats_set = set(category_filter)
        indices = [i for i in range(len(ds)) if (ds[i].get(cat_key) or "").strip() in cats_set]
        ds = ds.select(indices)

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
    cfg = BENCHMARK_CONFIGS[benchmark]
    cat_key = cfg.get("category_key")
    if cat_key and cat_key in example:
        return str(example[cat_key])
    return None
