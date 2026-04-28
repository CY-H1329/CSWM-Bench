#!/usr/bin/env python3
"""
Stage 2: WM frame generation scaffold.

Target WM: AdaWorld (Little-Podi/AdaWorld).

IMPORTANT:
- AdaWorld does not provide a simple "diffusers pipeline" for arbitrary images.
- This script is a **scaffold**:
  - It defines inputs/outputs and checkpointing.
  - It can call a local AdaWorld repo script if you provide --adaworld_repo and --adaworld_cmd.
  - If AdaWorld is not available, it falls back to a *no-op* (copies input frame) and logs.

Outputs:
  checkpoints/causalspatial_wm/wm_gen/<wm_key>/<split>/frames/<sample_id>/{t2,t3,t4}.png
  checkpoints/causalspatial_wm/wm_gen/<wm_key>/<split>/details.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

from scripts.evals.causalspatial_wm.load_causalspatial import load_all


ROOT = Path(__file__).resolve().parents[3]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_as_frame(src_img_path: str, out_dir: Path) -> List[str]:
    _ensure_dir(out_dir)
    out_paths = []
    for name in ("t2.png", "t3.png", "t4.png"):
        dst = out_dir / name
        shutil.copy2(src_img_path, dst)
        out_paths.append(dst.as_posix())
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_per_task", type=int, default=20)
    ap.add_argument("--checkpoints_dir", default=str(ROOT / "checkpoints" / "causalspatial_wm"))
    ap.add_argument("--wm_key", default="adaworld", choices=["adaworld", "cogvideox_fallback", "noop"])
    ap.add_argument("--adaworld_repo", default=None, help="Path to local AdaWorld repo (if available).")
    ap.add_argument(
        "--adaworld_cmd",
        default=None,
        help="Command template to run AdaWorld inference. Use {in_img} {out_dir}. Example: "
        "'python infer.py --image {in_img} --out {out_dir}'",
    )
    args = ap.parse_args()

    items = load_all(split=args.split, max_samples_per_task=args.max_per_task, checkpoints_dir=args.checkpoints_dir)

    run_dir = Path(args.checkpoints_dir) / "wm_gen" / args.wm_key / args.split / datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_root = run_dir / "frames"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    details = []
    for ex in tqdm(items, desc=f"wm_gen:{args.wm_key}"):
        sample_id = ex["sample_id"]
        in_img = ex["image_paths"][0] if ex["image_paths"] else ""
        out_dir = frames_root / sample_id
        _ensure_dir(out_dir)

        rec: Dict[str, Any] = {"sample_id": sample_id, "task_type": ex["task_type"], "wm_key": args.wm_key}

        try:
            if args.wm_key == "adaworld" and args.adaworld_repo and args.adaworld_cmd:
                # Execute user-provided AdaWorld inference command.
                cmd = args.adaworld_cmd.format(in_img=in_img, out_dir=out_dir.as_posix())
                subprocess.check_call(cmd, shell=True, cwd=str(Path(args.adaworld_repo).expanduser().resolve()))
                # Expect outputs exist; if not, fallback.
                outs = [str((out_dir / n).as_posix()) for n in ("t2.png", "t3.png", "t4.png") if (out_dir / n).exists()]
                if len(outs) != 3:
                    outs = _copy_as_frame(in_img, out_dir)
                    rec["warning"] = "adaworld_outputs_missing_fallback_to_copy"
                rec["wm_frames"] = outs
            else:
                # No-op fallback
                outs = _copy_as_frame(in_img, out_dir) if in_img else []
                rec["wm_frames"] = outs
                rec["warning"] = "wm_not_available_noop_copy"
        except Exception as e:
            rec["error"] = str(e)
            rec["wm_frames"] = _copy_as_frame(in_img, out_dir) if in_img else []

        details.append(rec)

    with open(run_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()

