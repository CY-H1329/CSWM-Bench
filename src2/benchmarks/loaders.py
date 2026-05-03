"""
Unified loaders for benchmarks (CV-Bench, 3DSRBench, ST-VQA, MindCube, …).
Returns normalized format: image, question, options (list or None), answer, category (optional).

3DSRBench: images are fetched from URL. Use image_cache_dir to cache locally for faster reruns.

Supports frozen benchmarks: when use_frozen=True (default), loads from data/frozen_benchmarks/
for reproducible paper experiments.
"""
import hashlib
import io
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk
from PIL import Image
import requests

# Frozen benchmark paths (DO NOT MODIFY - used for all paper experiments)
FROZEN_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "frozen_benchmarks"
FROZEN_PATHS = {
    "cvbench": "cvbench_400",
    "cvbench_counting_100": "cvbench_counting_100",  # scripts/create_cvbench_counting_100.py로 생성
    "3dsrbench": "3dsrbench_500",
    "stvqa": "stvqa_full",
}

# Local cache for 3DSRBench URL images (set to enable)
IMAGE_CACHE_DIR = os.environ.get("SPATIAL_MAS_IMAGE_CACHE")

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
    "cvbench_counting_100": {
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
    "stvqa": {
        "name": "hunarbatra/STVQA-7K",
        "split": "train",
        "image_key": "images",
        "question_key": "question_with_options",
        "question_fallback": "question_only",
        "options_key": "options",
        "answer_key": "answer",
        "answer_fallback": "answer_only",
        "category_key": "category",
    },
    # Hugging Face: zip ~600MB au premier chargement (H100 / cache HF recommandé).
    # Split: env MINDCUBE_SPLIT ou défaut ci-dessous (souvent train / test selon version HF).
    "mindcube": {
        "name": "MLL-Lab/MindCube",
        "split": "train",
        "image_key": "image",
        "question_key": "question",
        "options_key": "choices",
        "answer_key": "answer",
        "category_key": "task_type",
    },
}


def _url_to_cache_path(url: str) -> Optional[Path]:
    """Return cache path for URL if IMAGE_CACHE_DIR is set."""
    if not IMAGE_CACHE_DIR:
        return None
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = ".jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else ".png"
    return Path(IMAGE_CACHE_DIR) / f"{key}{ext}"


def _fetch_image_from_url(url: str) -> Optional[Image.Image]:
    """Fetch image from URL. Uses local cache if IMAGE_CACHE_DIR is set."""
    cache_path = _url_to_cache_path(url)
    if cache_path and cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if cache_path:
            Path(IMAGE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            img.save(cache_path, quality=95)
        return img
    except Exception:
        return None


def _mindcube_text_field(example: Dict, keys: tuple) -> str:
    for k in keys:
        v = example.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def mindcube_question(example: Dict) -> str:
    return _mindcube_text_field(
        example,
        ("question", "query", "instruction", "input_prompt", "problem", "text"),
    )


def mindcube_option_list(example: Dict) -> Optional[List[str]]:
    for k in ("choices", "options", "answer_choices", "candidates"):
        v = example.get(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out = [str(x).strip() for x in v if x is not None and str(x).strip()]
            return out or None
        if isinstance(v, str) and v.strip():
            parts = [p.strip() for p in v.split("|") if p.strip()]
            return parts or None
    return None


def mindcube_answer_raw(example: Dict) -> str:
    for k in ("answer", "label", "gold", "ground_truth", "gt"):
        v = example.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    v = example.get("grounded_output")
    if v is not None:
        s = str(v).strip()
        if s:
            return s
    return ""


def _mindcube_resize_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img.convert("RGB")
    if h == target_h:
        return img.convert("RGB")
    nw = max(1, int(w * (target_h / float(h))))
    return img.convert("RGB").resize((nw, target_h), Image.Resampling.LANCZOS)


def _mindcube_concat_horizontal(images: List[Image.Image]) -> Image.Image:
    target_h = min(im.height for im in images)
    resized = [_mindcube_resize_height(im, target_h) for im in images]
    total_w = sum(im.width for im in resized)
    out = Image.new("RGB", (total_w, target_h))
    x = 0
    for im in resized:
        out.paste(im, (x, 0))
        x += im.width
    return out


def mindcube_get_image(example: Dict) -> Optional[Image.Image]:
    """Une image ou plusieurs vues (liste) — collage horizontal pour le pipeline MAS."""
    for key in ("image", "images", "views", "pixel_values"):
        v = example.get(key)
        if v is None:
            continue
        if hasattr(v, "convert"):
            return v.convert("RGB")
        if isinstance(v, (list, tuple)):
            pil_list: List[Image.Image] = []
            for item in v[:8]:
                if item is None:
                    continue
                if hasattr(item, "convert"):
                    pil_list.append(item)
            if pil_list:
                return pil_list[0] if len(pil_list) == 1 else _mindcube_concat_horizontal(pil_list)
    for path_key in ("image_path", "img_path", "filepath"):
        p = example.get(path_key)
        if not p:
            continue
        root = os.environ.get("MINDCUBE_IMAGE_ROOT", "")
        path = Path(root) / str(p) if root else Path(str(p))
        if path.is_file():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                continue
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
    paper experiments. Applies max_samples/max_per_category if provided.

    When use_frozen=False: loads from HuggingFace cache and applies sampling.

    Returns dataset with normalized access via get_benchmark_* helpers.

    Args:
        category_filter: If set (and use_frozen=False), keep only samples whose category is in this list.
    """
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARK_CONFIGS.keys())}")

    ds = None
    # Try frozen benchmark first
    if use_frozen and benchmark in FROZEN_PATHS:
        frozen_name = FROZEN_PATHS[benchmark]
        frozen_path = FROZEN_BENCHMARK_DIR / frozen_name
        if frozen_path.exists() and (frozen_path / "dataset_info.json").exists():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(frozen_path))
            except (TypeError, Exception) as e:
                # datasets version mismatch (e.g. srgpt env vs spatial_reasoning)
                import warnings
                warnings.warn(
                    f"load_from_disk failed ({e}). Falling back to HuggingFace. "
                    "For reproducible frozen set, use matching datasets version."
                )

    # Fallback: load from HuggingFace if frozen didn't succeed
    cfg = dict(BENCHMARK_CONFIGS[benchmark])
    name = cfg["name"]
    split = cfg["split"]
    if benchmark == "mindcube":
        split = os.environ.get("MINDCUBE_SPLIT", split)
        cfg["split"] = split
    subset = cfg.get("subset")

    if ds is None:
        load_kw = {"split": split}

        def _try_load():
            if subset:
                return load_dataset(name, subset, **load_kw)
            return load_dataset(name, **load_kw)

        try:
            ds = _try_load()
        except (TypeError, Exception) as err1:
            err_str = str(err1)
            if "dataclass" in err_str or "must be called" in err_str:
                # datasets 2.16.1 cannot parse CV-Bench metadata from HF (created with newer datasets).
                # Fallback: load parquet directly (infers schema from file, bypasses dataset_info.json).
                if benchmark == "cvbench" and name == "nyu-visionx/CV-Bench":
                    try:
                        base = "https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/main"
                        data_files = {"test": [f"{base}/test_2d.parquet", f"{base}/test_3d.parquet"]}
                        ds = load_dataset("parquet", data_files=data_files, split="test")
                    except Exception as parquet_err:
                        raise RuntimeError(
                            f"CV-Bench load failed: standard ({err1}), parquet fallback ({parquet_err}). "
                            "Try: pip install datasets>=2.18 (may conflict with vila)"
                        ) from parquet_err
                else:
                    raise
            else:
                raise

    if ds is None:
        raise RuntimeError("load_benchmark failed")


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


def load_benchmark_from_dataset(
    benchmark: str,
    dataset_subdir: str,
    project_root: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    seed: int = 42,
):
    """
    Load benchmark from local data/dataset/<dataset_subdir> (e.g. 3dsrbench_train_300).
    Uses load_from_disk for datasets prepared by scripts/prepare_train_datasets.py.
    max_per_category: sample evenly across categories (e.g. 5 per category for 20 total on 4 cats).
    If both set: stratified by category first, then cap at max_samples.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    dataset_path = root / "data" / "dataset" / dataset_subdir
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run: git pull (or clone) and ensure data/dataset exists. "
            "Or: python scripts/prepare_train_datasets.py"
        )
    ds = load_from_disk(str(dataset_path))
    rng = random.Random(seed)
    cfg = BENCHMARK_CONFIGS.get(benchmark, {})
    cat_key = cfg.get("category_key")
    if max_per_category is not None and cat_key and cat_key in ds.features:
        by_cat = {}
        for i in range(len(ds)):
            c = (ds[i].get(cat_key) or "unknown").strip()
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            idx_list = by_cat[c]
            k = min(max_per_category, len(idx_list))
            indices.extend(rng.sample(idx_list, k))
        rng.shuffle(indices)
        if max_samples is not None and len(indices) > max_samples:
            indices = indices[:max_samples]
        ds = ds.select(indices)
    elif max_samples is not None and len(ds) > max_samples:
        indices = rng.sample(range(len(ds)), max_samples)
        indices.sort()
        ds = ds.select(indices)
    return ds


def get_benchmark_image(example: Dict, benchmark: str) -> Optional[Image.Image]:
    """Extract PIL Image from example."""
    if benchmark == "mindcube":
        img = mindcube_get_image(example)
        if img is not None:
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


def is_multiple_choice(example: Dict, benchmark: str) -> bool:
    """True if the question has multiple-choice options (A/B/C/D...), False for free-form/numeric."""
    if benchmark == "mindcube":
        return bool(mindcube_option_list(example))

    cfg = BENCHMARK_CONFIGS[benchmark]
    opts_key = cfg.get("options_key")
    opts_keys = cfg.get("options_keys")
    if opts_key and opts_key in example:
        opts = example[opts_key]
        return bool(opts and len(opts) > 0)
    if opts_keys:
        opts = [example.get(k) for k in opts_keys if example.get(k)]
        return bool(opts)
    return False


def infer_answer_type_from_query(query: str) -> str:
    """Infer 'multiple_choice' or 'free_form' from query string.
    Use when example is not available (e.g. pipeline only has query)."""
    if not query or not query.strip():
        return "free_form"
    q = query.strip().upper()
    if "OPTIONS:" in q and ("(A)" in q or "(B)" in q):
        return "multiple_choice"
    return "free_form"


def get_benchmark_prompt(example: Dict, benchmark: str, include_options: bool = True) -> str:
    """Build prompt (question + options if any)."""
    if benchmark == "mindcube":
        question = mindcube_question(example)
        if not include_options:
            return question
        opts = mindcube_option_list(example)
        if opts:
            lines = [question, "Options:"]
            for i, o in enumerate(opts):
                label = chr(65 + i)
                lines.append(f"({label}) {o}")
            return "\n".join(lines)
        return question

    cfg = BENCHMARK_CONFIGS[benchmark]
    q_key = cfg["question_key"]
    question = (example.get(q_key) or "").strip()
    if not question and cfg.get("question_fallback"):
        question = (example.get(cfg["question_fallback"]) or "").strip()

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
    if benchmark == "mindcube":
        s = mindcube_answer_raw(example)
        sup = s.upper()
        if mindcube_option_list(example):
            for c in "ABCDEF":
                if f"({c})" in sup or sup == c:
                    return c
            m = re.search(r"\b([A-F])\b", sup)
            if m:
                return m.group(1).upper()
        return s.strip()

    cfg = BENCHMARK_CONFIGS[benchmark]
    ans_key = cfg["answer_key"]
    ans = example.get(ans_key) or ""
    if not ans and cfg.get("answer_fallback"):
        ans = example.get(cfg["answer_fallback"]) or ""
    s = str(ans).strip()
    # Benchmarks with options: extract letter (CV-Bench uses "(C)")
    if cfg.get("options_key") or cfg.get("options_keys"):
        for c in "ABCDEF":
            if f"({c})" in s.upper() or s.upper() == c:
                return c
    return s


def get_benchmark_category(example: Dict, benchmark: str) -> Optional[str]:
    """Category if available (for evaluation only - never pass to Head-Agent)."""
    if benchmark == "mindcube":
        for k in ("task_type", "task", "category", "subtask", "split"):
            v = example.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return None

    cfg = BENCHMARK_CONFIGS[benchmark]
    cat_key = cfg.get("category_key")
    if cat_key and cat_key in example:
        return str(example[cat_key])
    return None
