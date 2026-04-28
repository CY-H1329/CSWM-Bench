#!/usr/bin/env python3
"""
Load CausalSpatial benchmark (arXiv:2601.13304) with robust fallbacks.

Primary source: HuggingFace dataset `Mwxinnn/CausalSpatial`.
Tasks/subsets: collision, occlusion, trajectory, compatibility.

Each sample is normalized to:
  {
    "sample_id": str,
    "task_type": str,
    "image_paths": [str],   # usually 1 image path on disk (cached)
    "question": str,
    "choices": {"A": str, "B": str, "C": str, "D": str},
    "answer": "A"|"B"|"C"|"D",
  }

If dataset is unavailable, falls back to a tiny manual set (for sanity runs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ID = "Mwxinnn/CausalSpatial"
TASK_TYPES = ("collision", "occlusion", "trajectory", "compatibility")


def _normalize_letter(x: str) -> str:
    s = (x or "").strip().upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else (s[:1] if s[:1] in "ABCD" else "")


def _ensure_choice_dict(raw: Any) -> Dict[str, str]:
    # raw can be list[str], dict, or four separate fields.
    if isinstance(raw, dict):
        out = {}
        for k in ("A", "B", "C", "D"):
            if k in raw:
                out[k] = str(raw[k])
        if len(out) == 4:
            return out
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        return {k: str(raw[i]) for i, k in enumerate(("A", "B", "C", "D"))}
    return {}


def _save_image_to_disk(image_obj: Any, out_dir: Path, sample_id: str) -> str:
    """
    HF datasets often return PIL images or dicts. We save to PNG for stable IO.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.png"
    if out_path.exists():
        return out_path.as_posix()

    img = None
    if hasattr(image_obj, "save"):
        img = image_obj
    elif isinstance(image_obj, dict) and "bytes" in image_obj:
        import io

        img = Image.open(io.BytesIO(image_obj["bytes"]))
    if img is None:
        raise ValueError("Unsupported image object")
    img = img.convert("RGB")
    img.save(out_path, format="PNG")
    return out_path.as_posix()


def load_causalspatial(
    task_type: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    checkpoints_dir: str = "checkpoints/causalspatial_wm",
) -> List[Dict[str, Any]]:
    task_type = task_type.lower().strip()
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type}")

    ckpt_root = (ROOT / checkpoints_dir).resolve()
    img_out_dir = ckpt_root / "cached_images" / task_type / split

    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, task_type, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        items: List[Dict[str, Any]] = []
        for i, ex in enumerate(ds):
            # Try common fields
            q = ex.get("question") or ex.get("query") or ex.get("prompt") or ""
            # choices might be in "choices"/"options"/"A..D"
            choices = _ensure_choice_dict(ex.get("choices") or ex.get("options") or ex.get("choice") or {})
            if not choices:
                cand = {k: ex.get(k) for k in ("A", "B", "C", "D")}
                if all(cand.get(k) is not None for k in ("A", "B", "C", "D")):
                    choices = {k: str(cand[k]) for k in ("A", "B", "C", "D")}

            ans = _normalize_letter(str(ex.get("answer") or ex.get("label") or ex.get("gt") or ""))

            # image field name
            img_field = None
            for k in ("image", "img", "image0", "rgb", "input_image"):
                if k in ex:
                    img_field = k
                    break
            if img_field is None:
                raise KeyError("No image field found in dataset example")
            sample_id = str(ex.get("id") or ex.get("sample_id") or f"{task_type}_{split}_{i:06d}")
            img_path = _save_image_to_disk(ex[img_field], img_out_dir, sample_id)

            items.append(
                {
                    "sample_id": sample_id,
                    "task_type": task_type,
                    "image_paths": [img_path],
                    "question": str(q),
                    "choices": choices,
                    "answer": ans,
                    "raw": {k: ex.get(k) for k in ex.keys() if k in ("id", "question", "answer", "choices", "A", "B", "C", "D")},
                }
            )
        return items
    except Exception:
        # Manual fallback: tiny set to allow pipeline testing.
        # NOTE: This is NOT the real benchmark.
        return [
            {
                "sample_id": f"manual_{task_type}_0001",
                "task_type": task_type,
                "image_paths": [],
                "question": f"[MANUAL FALLBACK] Example {task_type} question.",
                "choices": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                "answer": "A",
                "raw": {},
            }
        ]


def load_all(
    split: str = "train",
    max_samples_per_task: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in TASK_TYPES:
        out.extend(load_causalspatial(t, split=split, max_samples=max_samples_per_task, **kwargs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--task_type", choices=list(TASK_TYPES) + ["all"], default="all")
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "causalspatial_wm" / "dataset_preview.jsonl"))
    args = ap.parse_args()

    if args.task_type == "all":
        items = load_all(split=args.split, max_samples_per_task=args.max_samples)
    else:
        items = load_causalspatial(args.task_type, split=args.split, max_samples=args.max_samples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()

