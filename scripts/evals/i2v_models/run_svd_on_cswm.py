#!/usr/bin/env python3
"""
Run Stable Video Diffusion (SVD) on CSWM image prompts.

This is a qualitative/PoC runner:
  - loads CSWM items from data/cswmbench/cswmbench.jsonl
  - for Task A: generates a video for each image (case1/case2) with a simple prompt
  - for Task B: generates two videos from the same image with two prompts (push 2cm vs push 6cm)

Outputs MP4 videos in:
  results/runs/i2v/svd/<timestamp>/videos/
and a machine-readable manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_image(rel_path: str) -> Image.Image:
    return Image.open((ROOT / rel_path).resolve()).convert("RGB")


def _save_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "cswmbench" / "cswmbench.jsonl"))
    ap.add_argument("--model_id", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--max_items", type=int, default=20)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_frames", type=int, default=14)
    ap.add_argument("--motion_bucket_id", type=int, default=127)
    ap.add_argument("--noise_aug_strength", type=float, default=0.02)
    args = ap.parse_args()

    # Lazy import (only when actually running)
    import torch
    from diffusers import StableVideoDiffusionPipeline

    out_dir = ROOT / "results" / "runs" / "i2v" / "svd" / datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # symlink-ish "latest" pointer folder (best effort)
    latest = ROOT / "results" / "runs" / "i2v" / "svd" / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                # if folder, remove marker file only
                pass
        # Can't reliably symlink on all FS; write a pointer file instead.
        (latest.parent / "LATEST.txt").write_text(out_dir.as_posix(), encoding="utf-8")
    except Exception:
        pass

    items = _read_jsonl(Path(args.data))
    items = items[: min(args.max_items, len(items))]

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(args.device)
    pipe.enable_model_cpu_offload() if args.device == "cuda" else None

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for ex in tqdm(items, desc="svd"):
            ex_id = ex.get("id")
            task = ex.get("task")
            imgs = ex.get("images", []) or []

            jobs = []
            if task == "A" and len(imgs) == 2:
                jobs = [
                    {"key": "case1", "image": imgs[0], "prompt": "A door opens to 90 degrees."},
                    {"key": "case2", "image": imgs[1], "prompt": "A door opens to 90 degrees."},
                ]
            elif task == "B" and len(imgs) == 2:
                # Same action prompt; only the image differs slightly.
                jobs = [
                    {"key": "case1", "image": imgs[0], "prompt": "A hand pushes the cup 4cm to the right."},
                    {"key": "case2", "image": imgs[1], "prompt": "A hand pushes the cup 4cm to the right."},
                ]
            else:
                continue

            for j in jobs:
                im = _load_image(j["image"])
                # SVD is image-conditioned only (no text). Keep prompt in manifest only.
                gen = pipe(
                    image=im,
                    num_frames=args.num_frames,
                    motion_bucket_id=args.motion_bucket_id,
                    noise_aug_strength=args.noise_aug_strength,
                    generator=torch.Generator(device=args.device).manual_seed(args.seed),
                )
                frames = gen.frames[0]  # list of PIL
                frames_np = [np.asarray(fr.convert("RGB")) for fr in frames]
                out_path = videos_dir / f"{ex_id}_{j['key']}.mp4"
                _save_mp4(frames_np, out_path, fps=args.fps)

                rec = {
                    "id": ex_id,
                    "task": task,
                    "job": j["key"],
                    "image": j["image"],
                    "prompt": j["prompt"],
                    "video": out_path.relative_to(ROOT).as_posix(),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote: {manifest_path}")
    print(f"Videos: {videos_dir}")
    print(f"Run dir: {out_dir}")


if __name__ == "__main__":
    main()

