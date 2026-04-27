#!/usr/bin/env python3
"""
Cosmos 1.0 (Video2World) custom runner for user-provided images.

Generates one video per (image, prompt). This matches your intended test:
- Door: 2 images, SAME prompt "Open the door."
- Cup: 1 image, 2 prompts "Push 5cm" vs "Push 30cm"

Uses Diffusers CosmosVideoToWorldPipeline when available.

Outputs:
  results/runs/i2v/cosmos_custom/<timestamp>/
    inputs/
    videos/
    manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True, help="Path to an input image (repeatable)")
    ap.add_argument("--prompt", action="append", required=True, help="Text prompt/action (repeatable)")
    ap.add_argument("--group", default="custom", help="Group name for filenames (door/cup)")
    ap.add_argument(
        "--model_id",
        default="nvidia/Cosmos-1.0-Diffusion-7B-Video2World",
        help="HF model id (Video2World). 14B is heavier: nvidia/Cosmos-1.0-Diffusion-14B-Video2World",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=121, help="Cosmos default: 121 frames (5s at 24fps)")
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    import torch
    from diffusers import CosmosVideoToWorldPipeline
    from diffusers.utils import export_to_video

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    run_dir = ROOT / "results" / "runs" / "i2v" / "cosmos_custom" / datetime.now().strftime("%Y%m%d_%H%M%S")
    inputs_dir = run_dir / "inputs"
    videos_dir = run_dir / "videos"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Copy inputs for portability
    input_files: List[Path] = []
    for p in args.image:
        src = Path(p).expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"Missing image: {src}")
        dst = inputs_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        input_files.append(dst)

    pipe = CosmosVideoToWorldPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(args.device)

    g = torch.Generator(device=args.device).manual_seed(args.seed)

    manifest_path = run_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for im_path in tqdm(input_files, desc="images"):
            im = Image.open(im_path).convert("RGB")
            # Resize to expected aspect (Cosmos supports multiple aspect ratios; keep yours consistent)
            im = im.resize((args.width, args.height), Image.Resampling.LANCZOS)
            for j, prompt in enumerate(args.prompt):
                frames = pipe(image=im, prompt=prompt, generator=g, num_frames=args.num_frames).frames[0]
                out_path = videos_dir / f"{args.group}_{im_path.stem}_p{j+1}.mp4"
                export_to_video(frames, str(out_path), fps=args.fps)

                mf.write(
                    json.dumps(
                        {
                            "id": f"{args.group}:{im_path.stem}",
                            "task": args.group,
                            "job": f"prompt{j+1}",
                            "image": str(im_path.relative_to(ROOT)),
                            "prompt": prompt,
                            "video": str(out_path.relative_to(ROOT)),
                            "model_id": args.model_id,
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

