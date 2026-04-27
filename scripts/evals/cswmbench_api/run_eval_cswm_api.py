#!/usr/bin/env python3
"""
CSWM-Bench (toy) — API evaluation with divergence metrics.

This is intentionally standalone (does not modify src/benchmarks).
It reuses the existing API runners from scripts/evals/3dsrbench_api/runners.py.

Usage:
  python scripts/evals/cswmbench_api/generate_cswmbench.py
  python scripts/evals/cswmbench_api/run_eval_cswm_api.py --model gpt4o
  python scripts/evals/cswmbench_api/run_eval_cswm_api.py --model qwen3_vl_32b_instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load_runners():
    import importlib.util

    runners_path = ROOT / "scripts/evals/3dsrbench_api/runners.py"
    spec = importlib.util.spec_from_file_location("cswm_runners", runners_path)
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
    ims: List[Image.Image] = []
    for rp in rel_paths:
        p = (ROOT / rp).resolve()
        ims.append(Image.open(p).convert("RGB"))
    return ims


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction:
    - prefer the first {...} block
    - fallback to empty dict
    """
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        # Try a minimal cleanup (trailing commas)
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except Exception:
            return {}


def _score_item(item: Dict[str, Any], pred_obj: Dict[str, Any]) -> Dict[str, Any]:
    task = item.get("task")
    gt = item.get("gt", {}) or {}
    out: Dict[str, Any] = {"task": task, "id": item.get("id")}

    if task == "A":
        gt_div = gt.get("divergence")
        gt_c1 = gt.get("case1_outcome")
        gt_c2 = gt.get("case2_outcome")
        gt_reason = gt.get("reason_label")

        pr_div = str(pred_obj.get("divergence", "")).strip().lower()
        pr_c1 = str(pred_obj.get("case1_outcome", "")).strip().lower()
        pr_c2 = str(pred_obj.get("case2_outcome", "")).strip().lower()
        pr_reason = str(pred_obj.get("reason_label", "")).strip()

        out["divergence_correct"] = (pr_div == str(gt_div).strip().lower())
        out["outcomes_correct"] = (pr_c1 == str(gt_c1).strip().lower() and pr_c2 == str(gt_c2).strip().lower())
        out["reason_correct"] = (pr_reason == gt_reason)
        out["all_correct"] = out["divergence_correct"] and out["outcomes_correct"] and out["reason_correct"]
        return out

    if task == "B":
        gt_label = gt.get("divergence_label")
        gt_reason = gt.get("reason_label")

        pr_label = str(pred_obj.get("divergence_label", "")).strip()
        pr_reason = str(pred_obj.get("reason_label", "")).strip()

        out["divergence_correct"] = (pr_label == gt_label)
        out["reason_correct"] = (pr_reason == gt_reason)
        out["all_correct"] = out["divergence_correct"] and out["reason_correct"]
        return out

    out["error"] = f"unknown_task:{task}"
    return out


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    def safe_mean(xs: List[bool]) -> float:
        if not xs:
            return 0.0
        return sum(1 for x in xs if x) / len(xs)

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in scores:
        by_task[s.get("task", "unknown")].append(s)

    overall = {
        "n": len(scores),
        "divergence_accuracy": safe_mean([s.get("divergence_correct", False) for s in scores]),
        "reason_accuracy": safe_mean([s.get("reason_correct", False) for s in scores]),
        "all_correct": safe_mean([s.get("all_correct", False) for s in scores]),
    }

    per_task = {}
    for t, rows in sorted(by_task.items()):
        per_task[t] = {
            "n": len(rows),
            "divergence_accuracy": safe_mean([r.get("divergence_correct", False) for r in rows]),
            "reason_accuracy": safe_mean([r.get("reason_correct", False) for r in rows]),
            "all_correct": safe_mean([r.get("all_correct", False) for r in rows]),
        }
    return {"overall": overall, "per_task": per_task}


def get_runner(model_key: str, config: dict):
    cfg = config.get("models", {}).get(model_key, {})
    if not cfg.get("enabled", True):
        return None

    # OpenAI-compatible (including OpenRouter)
    if cfg.get("api_runner") == "openai_compatible":
        api_key = os.environ.get(cfg.get("api_key_env", ""), "").strip()
        if not api_key:
            return None
        base_url = cfg.get("base_url", "").strip()
        if not base_url:
            # If omitted, use OpenAI default inside OpenAI client
            return _RUNNERS.GPT4oRunner(model_id=cfg.get("model_id", ""), api_key=api_key)
        return _RUNNERS.OpenRouterRunner(
            model_id=cfg.get("model_id", ""),
            api_key=api_key,
            base_url=base_url,
            text_only=bool(cfg.get("text_only", False)),
        )

    # Reuse existing keys
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
    parser = argparse.ArgumentParser(description="CSWM-Bench (toy) — API eval")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_cswm_api.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "cswmbench" / "cswmbench.jsonl"))
    parser.add_argument(
        "--model",
        default="gpt4o",
        help="Model key from config (e.g., gpt4o, gpt_5_2, gemini_robotics_er, qwen3_vl_32b_instruct)",
    )
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="No API call; emits empty predictions to validate the pipeline.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runner = None if args.dry_run else get_runner(args.model, config)
    if not args.dry_run and runner is None:
        print(f"[ERREUR] Runner indisponible pour model={args.model} (clé API manquante ou disabled).")
        sys.exit(2)

    data = _read_jsonl(Path(args.data))
    if not data:
        print(f"[ERREUR] dataset vide: {args.data}. Lance d'abord generate_cswmbench.py")
        sys.exit(2)

    n = min(args.max_samples, len(data)) if args.max_samples else len(data)
    data = data[:n]

    out_root = Path(config.get("output", {}).get("dir", "results"))
    run_dir = out_root / "runs" / "cswmbench" / "api_models" / datetime.now().strftime("%Y%m%d_%H%M%S") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)

    details: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []

    for ex in tqdm(data, desc=f"cswmbench:{args.model}"):
        ims = _load_images(ex.get("images", []))
        prompt = ex.get("prompt", "")
        payload = ims[0] if len(ims) == 1 else ims

        try:
            if args.dry_run:
                resp = "{}"
            else:
                resp = runner.generate(payload, prompt, temperature=0.0, max_tokens=args.max_tokens)
        except Exception as e:
            resp = ""
            err = str(e)
        else:
            err = ""

        pred_obj = _extract_json(resp)
        sc = _score_item(ex, pred_obj)
        scores.append(sc)

        d = {
            "id": ex.get("id"),
            "task": ex.get("task"),
            "category": ex.get("category"),
            "images": ex.get("images"),
            "gt": ex.get("gt"),
            "prompt": ex.get("prompt"),
            "response": resp,
            "pred_obj": pred_obj,
            "score": sc,
        }
        if err:
            d["error"] = err
        details.append(d)

        with open(run_dir / "responses" / f"{ex.get('id')}.txt", "w", encoding="utf-8") as f:
            f.write(resp or "")

    metrics = _aggregate(scores)
    with open(run_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Résultats: {run_dir}")


if __name__ == "__main__":
    main()

