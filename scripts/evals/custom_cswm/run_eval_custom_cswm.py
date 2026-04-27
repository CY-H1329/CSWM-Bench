#!/usr/bin/env python3
"""
Run a quick qualitative QCM eval on the custom 2-task dataset.

This intentionally does NOT auto-score by default (GT is left blank).
It writes details.jsonl with model outputs so YOU can compare.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load_runners():
    import importlib.util

    runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
    spec = importlib.util.spec_from_file_location("custom_runners", runners_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RUNNERS = _load_runners()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_images(rel_paths: List[str]) -> List[Image.Image]:
    ims = []
    for rp in rel_paths:
        p = (ROOT / rp).resolve()
        ims.append(Image.open(p).convert("RGB"))
    return ims


def _extract_letter(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def get_runner(model_key: str, config: dict):
    cfg = config.get("models", {}).get(model_key, {})
    if not cfg.get("enabled", True):
        return None
    api_key = os.environ.get(cfg.get("api_key_env", ""), "").strip()
    if not api_key:
        return None
    model_id = cfg.get("model_id", "")
    if model_key == "claude_sonnet_4_5":
        return _RUNNERS.ClaudeRunner(model_id=model_id, api_key=api_key)
    if model_key in ("gpt4o", "gpt_5_2"):
        return _RUNNERS.GPT4oRunner(model_id=model_id, api_key=api_key)
    if model_key == "gemini_robotics_er":
        return _RUNNERS.GeminiRunner(model_id=model_id, api_key=api_key)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config_custom_eval.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data" / "custom_cswm" / "custom_cswm.jsonl"))
    ap.add_argument("--model", default="gpt4o")
    ap.add_argument("--max_samples", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runner = get_runner(args.model, config)
    if runner is None:
        raise SystemExit(f"Runner unavailable for model={args.model} (missing API key or disabled).")

    data = _read_jsonl(Path(args.data))
    data = data[: min(args.max_samples, len(data))]

    out_root = Path(config.get("output", {}).get("dir", "results"))
    run_dir = out_root / "runs" / "custom_cswm" / datetime.now().strftime("%Y%m%d_%H%M%S") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    details = []
    for ex in tqdm(data, desc=f"custom_cswm:{args.model}"):
        ims = _load_images(ex.get("images", []))
        payload = ims[0] if len(ims) == 1 else ims
        prompt = ex.get("prompt", "")
        resp = runner.generate(payload, prompt, temperature=0.0, max_tokens=args.max_tokens)
        pred = _extract_letter(resp)
        details.append(
            {
                "id": ex.get("id"),
                "task": ex.get("task"),
                "images": ex.get("images"),
                "prompt": ex.get("prompt"),
                "response": resp,
                "pred_letter": pred,
            }
        )

    with open(run_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()

