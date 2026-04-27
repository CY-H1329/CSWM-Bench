#!/usr/bin/env python3
"""
Custom qualitative I2V runner (SVD) for user-provided images.

This is designed for "human-in-the-loop" comparison (no automatic scoring):
- You provide one or more images
- You provide one or more action prompts (text)
- It generates one video per (image, prompt)

Outputs:
  results/runs/i2v/svd_custom/<timestamp>/
    inputs/   (copied images)
    videos/   (*.mp4)
    manifest.jsonl  (for make_video_viewer.py)

Example:
  conda activate i2v
  python scripts/evals/i2v_models/run_svd_custom.py \
    --image "/path/door_empty.png" --image "/path/door_box.png" \
    --prompt "Open the door." \
    --group door

  python scripts/evals/i2v_models/run_svd_custom.py \
    --image "/path/cup.png" \
    --prompt "Move the cup 5cm to the right." \
    --prompt "Move the cup 30cm to the right." \
    --group cup
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]


def _save_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True, help="Path to an input image (can repeat)")
    ap.add_argument("--prompt", action="append", required=True, help="Action prompt text (can repeat)")
    ap.add_argument("--group", default="custom", help="Group name (door/cup/...) for filenames")
    ap.add_argument("--model_id", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_frames", type=int, default=14)
    ap.add_argument("--motion_bucket_id", type=int, default=127)
    ap.add_argument("--noise_aug_strength", type=float, default=0.02)
    args = ap.parse_args()

    import torch
    from diffusers import StableVideoDiffusionPipeline

    run_dir = ROOT / "results" / "runs" / "i2v" / "svd_custom" / datetime.now().strftime("%Y%m%d_%H%M%S")
    inputs_dir = run_dir / "inputs"
    videos_dir = run_dir / "videos"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Copy inputs for portability
    input_files = []
    for p in args.image:
        src = Path(p).expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"Missing image: {src}")
        dst = inputs_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        input_files.append(dst)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(args.device)
    pipe.enable_model_cpu_offload() if args.device == "cuda" else None

    manifest_path = run_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for im_path in tqdm(input_files, desc="images"):
            im = Image.open(im_path).convert("RGB")
            for j, prompt in enumerate(args.prompt):
                job_key = f"{args.group}_p{j+1}"
                gen = pipe(
                    image=im,
                    prompt=prompt,
                    num_frames=args.num_frames,
                    motion_bucket_id=args.motion_bucket_id,
                    noise_aug_strength=args.noise_aug_strength,
                    generator=torch.Generator(device=args.device).manual_seed(args.seed),
                )
                frames = gen.frames[0]
                frames_np = [np.asarray(fr.convert("RGB")) for fr in frames]
                out_path = videos_dir / f"{args.group}_{im_path.stem}_p{j+1}.mp4"
                _save_mp4(frames_np, out_path, fps=args.fps)

                mf.write(
                    json.dumps(
                        {
                            "id": f"{args.group}:{im_path.stem}",
                            "task": args.group,
                            "job": f"prompt{j+1}",
                            "image": str(im_path.relative_to(ROOT)),
                            "prompt": prompt,
                            "video": str(out_path.relative_to(ROOT)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print("Next:")
    print(f"  python scripts/evals/i2v_models/make_video_viewer.py --run_dir {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

