#!/usr/bin/env python3
"""
Compute embedding separation on CSWM image pairs.

Default encoder = CLIP ViT-B/32 (open, easy) to validate the pipeline.
You can plug in VL-JEPA via --encoder custom + --custom_loader.

Outputs:
  <out>/<timestamp>/results.json
  <out>/<timestamp>/pairs.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_image(rel_path: str) -> Image.Image:
    return Image.open((ROOT / rel_path).resolve()).convert("RGB")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _default_clip_encoder(device: str) -> Callable[[Image.Image], np.ndarray]:
    """
    Uses OpenAI CLIP via transformers. This is NOT VL-JEPA,
    but it gives you a working baseline immediately.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    @torch.inference_mode()
    def encode(im: Image.Image) -> np.ndarray:
        inputs = proc(images=im, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        v = feats[0].detach().float().cpu().numpy()
        return v

    return encode


def _custom_encoder(custom_loader_path: str, device: str) -> Callable[[Image.Image], np.ndarray]:
    """
    custom_loader_path must define:
      def load_encoder(device: str) -> Callable[[PIL.Image], np.ndarray]:
          ...
    """
    p = Path(custom_loader_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("custom_vljepa_loader", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_encoder"):
        raise SystemExit("custom loader must define load_encoder(device) -> encode(image)->np.ndarray")
    return mod.load_encoder(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with items containing images (len=2)")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs" / "vl_jepa_embed"))
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--encoder", choices=["clip", "custom"], default="clip")
    ap.add_argument("--custom_loader", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    items = _read_jsonl(Path(args.data))
    pairs = [it for it in items if isinstance(it.get("images"), list) and len(it["images"]) == 2]
    pairs = pairs[: min(args.max_pairs, len(pairs))]

    if args.encoder == "clip":
        encode = _default_clip_encoder(args.device)
        enc_name = "clip_vit_b32"
    else:
        if not args.custom_loader:
            raise SystemExit("--custom_loader is required for encoder=custom")
        encode = _custom_encoder(args.custom_loader, args.device)
        enc_name = Path(args.custom_loader).stem

    out_root = Path(args.out)
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dists = []
    for it in tqdm(pairs, desc=f"embed:{enc_name}"):
        p1, p2 = it["images"][0], it["images"][1]
        im1 = _load_image(p1)
        im2 = _load_image(p2)
        e1 = encode(im1)
        e2 = encode(im2)
        cos = _cosine(e1, e2)
        dist = 1.0 - cos
        dists.append(dist)
        rows.append(
            {
                "id": it.get("id"),
                "task": it.get("task"),
                "category": it.get("category"),
                "image1": p1,
                "image2": p2,
                "cosine": cos,
                "distance_1_minus_cos": dist,
                "gt": it.get("gt", {}),
            }
        )

    # Summary stats
    arr = np.asarray(dists, dtype=np.float32) if dists else np.zeros((0,), dtype=np.float32)
    summary = {
        "encoder": enc_name,
        "n_pairs": int(len(rows)),
        "distance_mean": float(arr.mean()) if len(arr) else 0.0,
        "distance_median": float(np.median(arr)) if len(arr) else 0.0,
        "distance_p90": float(np.quantile(arr, 0.9)) if len(arr) else 0.0,
    }

    with open(run_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()

