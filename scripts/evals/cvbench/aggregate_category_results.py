#!/usr/bin/env python3
"""
CV-Bench — 결과를 category별, with_prompt / without_prompt로 정리.

Lit les details.jsonl, calcule l'accuracy par catégorie, sépare with/without prompt.
Output: CSV + JSON organisés par category et prompt variant.

Usage:
  python scripts/evals/cvbench/aggregate_category_results.py --dir results/runs/cvbench
  python scripts/evals/cvbench/aggregate_category_results.py --dir results/runs/cvbench --output results/analysis/cvbench_by_category.csv
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# CV-Bench 4 categories (canonical order)
CVBENCH_CATEGORIES = ["Count", "Relation", "Depth", "Distance"]


def _normalize_category(cat: str) -> str:
    """Normalize to canonical: Count, Relation, Depth, Distance."""
    if not cat:
        return ""
    s = str(cat).strip().lower().replace(" ", "_").replace("-", "_")
    for c in CVBENCH_CATEGORIES:
        if c.lower().replace(" ", "_") == s:
            return c
    return cat  # return as-is if unknown


def load_details(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_category_from_responses(responses_dir: Path) -> Dict[int, str]:
    """Parse responses/sample_*.txt for category_gt."""
    out = {}
    for p in sorted(responses_dir.glob("sample_*.txt")):
        try:
            m = re.search(r"sample_(\d+)\.txt", p.name)
            idx = int(m.group(1)) if m else -1
            text = p.read_text(encoding="utf-8")
            match = re.search(r"=== CATEGORY GT / PRED ===\s*\n\s*([^\n]+)", text)
            if match:
                line = match.group(1).strip()
                cat = line.split(" /", 1)[0].strip() if " /" in line else line.split(" / ", 1)[0].strip()
                if idx >= 0 and cat:
                    out[idx] = _normalize_category(cat)
        except Exception:
            pass
    return out


def enrich_details(details: List[dict], details_path: Path) -> List[dict]:
    """Fill missing category_gt from responses/."""
    need = any(not _normalize_category(d.get("category_gt", "") or d.get("category", "")) for d in details)
    if not need:
        return details
    resp_dir = details_path.parent / "responses"
    if not resp_dir.exists():
        resp_dir = details_path.parent.parent / "responses"
    idx2cat = _parse_category_from_responses(resp_dir) if resp_dir.exists() else {}
    for d in details:
        if not _normalize_category(d.get("category_gt", "") or d.get("category", "")):
            idx = d.get("idx", -1)
            if idx in idx2cat:
                d["category_gt"] = idx2cat[idx]
    return details


def _norm_answer(s: str) -> str:
    """Normalize answer for comparison (A, (A), (A) -> A)."""
    if not s:
        return ""
    s = str(s).strip().upper()
    for c in "ABCDEF":
        if c in s or f"({c})" in s:
            return c
    return s


def compute_per_category(details: List[dict]) -> Dict[str, dict]:
    """Returns {category: {correct, total, accuracy}}."""
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in details:
        cat_raw = d.get("category_gt", "") or d.get("category", "")
        cat = _normalize_category(cat_raw)
        if not cat:
            continue
        gt = _norm_answer(d.get("gt", ""))
        pred = _norm_answer(d.get("pred", ""))
        by_cat[cat]["total"] += 1
        if pred and pred == gt:
            by_cat[cat]["correct"] += 1
    result = {}
    for cat in CVBENCH_CATEGORIES:
        if cat in by_cat:
            v = by_cat[cat]
            result[cat] = {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
    for cat, v in by_cat.items():
        if cat not in result:
            result[cat] = {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
    return result


def find_cvbench_runs(base: Path, full_dataset_only: bool = True) -> List[Tuple[str, str, Path]]:
    """
    Find (model, prompt_variant, details_path).
    full_dataset_only: only full_dataset_with_prompt, full_dataset_without_prompt, api_models/full_dataset
    """
    runs = []
    base = Path(base).resolve()

    # GPU: llava4d/, qwen3_4b/, sa2va/ → full_dataset_with_prompt, full_dataset_without_prompt only
    for model_dir in ["llava4d", "qwen3_4b", "sa2va"]:
        model_path = base / model_dir
        if not model_path.is_dir():
            continue
        for sub in model_path.iterdir():
            if not sub.is_dir():
                continue
            if full_dataset_only and "full_dataset" not in sub.name:
                continue
            details_path = sub / "details.jsonl"
            if not details_path.exists():
                continue
            name = sub.name
            if "with_prompt" in name or "with prompt" in name.lower():
                variant = "with_prompt"
            elif "without_prompt" in name or "without prompt" in name.lower():
                variant = "without_prompt"
            else:
                variant = name
            runs.append((model_dir, variant, details_path))

    # API: api_models/full_dataset/ only (skip timestamp dirs)
    api_base = base / "api_models"
    if api_base.exists():
        # Prefer full_dataset dir
        full_dir = api_base / "full_dataset"
        ts_dirs = [full_dir] if full_dir.exists() else []
        if not ts_dirs:
            ts_dirs = [d for d in api_base.iterdir() if d.is_dir()]
        for ts_dir in ts_dirs:
            if full_dataset_only and ts_dir.name != "full_dataset":
                continue
            for sub in ts_dir.iterdir():
                if not sub.is_dir():
                    continue
                details_path = sub / "details.jsonl"
                if not details_path.exists():
                    continue
                name = sub.name
                if "_with_prompt" in name:
                    variant = "with_prompt"
                elif "_without_prompt" in name:
                    variant = "without_prompt"
                else:
                    variant = "unknown"
                model = name.replace("_with_prompt", "").replace("_without_prompt", "").strip("_")
                runs.append((model, variant, details_path))

    # Auto: recursive (only if no runs and full_dataset_only=False)
    if not runs and not full_dataset_only:
        for details_path in base.rglob("details.jsonl"):
            rel = details_path.relative_to(base)
            parts = list(rel.parts[:-1])
            name = "_".join(parts) if parts else details_path.parent.name
            if "with_prompt" in name or "with_prompt" in str(details_path):
                variant = "with_prompt"
            elif "without_prompt" in name or "without_prompt" in str(details_path):
                variant = "without_prompt"
            else:
                variant = "unknown"
            model = parts[0] if parts else details_path.parent.name
            runs.append((model, variant, details_path))

    return runs


def main():
    parser = argparse.ArgumentParser(description="CV-Bench — Résultats par catégorie, with/without prompt")
    parser.add_argument("--dir", default="results/runs/cvbench", help="Répertoire racine cvbench")
    parser.add_argument("--output", default=None, help="Fichier CSV de sortie")
    parser.add_argument("--output_dir", default=None, help="Répertoire de sortie (défaut: --dir)")
    parser.add_argument("--all_runs", action="store_true", help="Inclure tous les runs (défaut: full_dataset uniquement)")
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"[ERREUR] --dir n'existe pas: {base}")
        return 1

    runs = find_cvbench_runs(base, full_dataset_only=not args.all_runs)
    if not runs:
        print(f"[ERREUR] Aucun details.jsonl trouvé sous {base}")
        return 1

    print(f"Trouvé {len(runs)} runs:")
    for model, variant, path in runs:
        n = sum(1 for d in load_details(path) if _normalize_category(d.get("category_gt") or d.get("category")))
        print(f"  - {model} | {variant} ({n} samples)")

    # Compute per category for each run
    all_results = {}
    for model, variant, details_path in runs:
        key = f"{model}_{variant}"
        details = load_details(details_path)
        details = enrich_details(details, details_path)
        per_cat = compute_per_category(details)
        all_results[key] = {
            "model": model,
            "prompt_variant": variant,
            "per_category": per_cat,
            "overall": {
                "correct": sum(v["correct"] for v in per_cat.values()),
                "total": sum(v["total"] for v in per_cat.values()),
            },
        }
        if all_results[key]["overall"]["total"]:
            all_results[key]["overall"]["accuracy"] = (
                all_results[key]["overall"]["correct"] / all_results[key]["overall"]["total"]
            )
        else:
            all_results[key]["overall"]["accuracy"] = 0.0

    out_dir = Path(args.output_dir) if args.output_dir else base
    out_csv = Path(args.output) if args.output else out_dir / "cvbench_by_category.csv"
    out_json = out_dir / "cvbench_by_category.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: model, prompt_variant, category, accuracy, correct, total
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("model,prompt_variant,category,accuracy,correct,total\n")
        for key, data in all_results.items():
            for cat in CVBENCH_CATEGORIES:
                if cat in data["per_category"]:
                    v = data["per_category"][cat]
                    f.write(f"{data['model']},{data['prompt_variant']},{cat},{v['accuracy']:.4f},{v['correct']},{v['total']}\n")
            for cat, v in data["per_category"].items():
                if cat not in CVBENCH_CATEGORIES:
                    f.write(f"{data['model']},{data['prompt_variant']},{cat},{v['accuracy']:.4f},{v['correct']},{v['total']}\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary tables: by category, then by with/without prompt
    print("\n" + "=" * 80)
    print("CV-Bench — Performance par catégorie (with_prompt | without_prompt)")
    print("=" * 80)

    models = sorted(set(r["model"] for r in all_results.values()))
    variants = ["with_prompt", "without_prompt"]

    for cat in CVBENCH_CATEGORIES:
        print(f"\n--- {cat} ---")
        header = "model".ljust(20) + "".join(v.ljust(22) for v in variants)
        print(header)
        print("-" * len(header))
        for model in models:
            line = model[:18].ljust(20)
            for v in variants:
                key = f"{model}_{v}"
                if key in all_results and cat in all_results[key]["per_category"]:
                    val = all_results[key]["per_category"][cat]
                    line += f"{val['accuracy']:.2f} ({val['total']})".ljust(22)
                else:
                    line += "-".ljust(22)
            print(line)

    print("\n" + "=" * 80)
    print("Overall")
    print("-" * 80)
    header = "model".ljust(20) + "".join(v.ljust(22) for v in variants)
    print(header)
    for model in models:
        line = model[:18].ljust(20)
        for v in variants:
            key = f"{model}_{v}"
            if key in all_results:
                val = all_results[key]["overall"]
                line += f"{val['accuracy']:.2f} ({val['total']})".ljust(22)
            else:
                line += "-".ljust(22)
        print(line)

    print("=" * 80)
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    return 0


if __name__ == "__main__":
    exit(main())
