#!/usr/bin/env python3
"""
3DSRBench — Agrégation des résultats par catégorie (6 modèles × 2 variants).

Lit les details.jsonl de chaque run, calcule l'accuracy par catégorie, sauvegarde CSV + JSON.

Usage:
  # API (un run timestamp ou full_dataset)
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench/api_models/20260217_023920

  # API full_dataset
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench/api_models/full_dataset

  # GPU (qwen3_4b, llava4d, sa2va avec full_dataset_with/without_prompt)
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --mode gpu

  # GPU — 6 runs exacts (éditer runs_gpu_6.json pour les chemins)
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --runs_file scripts/evals/3dsrbench/runs_gpu_6.json

  # Auto (trouve tous les details.jsonl récursivement)
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --mode auto

  # Sortie personnalisée
  python scripts/evals/3dsrbench/aggregate_category_performance.py --dir ... --output category_results.csv
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 12 catégories 3DSRBench (ordre canonique)
CATEGORIES = [
    "location_above",
    "height_higher",
    "location_closer_to_camera",
    "multi_object_closer_to",
    "orientation_on_the_left",
    "multi_object_facing",
    "multi_object_same_direction",
    "orientation_in_front_of",
    "multi_object_viewpoint_towards_object",
    "orientation_viewpoint",
    "location_next_to",
    "multi_object_parallel",
]


def load_details(path: Path) -> List[dict]:
    """Load details.jsonl, return list of records."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _get_idx_to_category(seed: int = 42) -> Optional[dict]:
    """Load 3DSRBench and return {idx: normalized_category}. Lazy, cached."""
    if not hasattr(_get_idx_to_category, "_cache"):
        try:
            from src.benchmarks import load_benchmark, get_benchmark_category
            from src.data import normalize_category
            ds = load_benchmark("3dsrbench", max_samples=None, seed=seed)
            out = {}
            for i in range(len(ds)):
                ex = ds[i]
                cat = get_benchmark_category(ex, "3dsrbench") or "unknown"
                norm = normalize_category(cat) if cat and cat != "unknown" else ""
                out[i] = norm
            _get_idx_to_category._cache = out
        except Exception as e:
            print(f"[!] Impossible de charger 3DSRBench pour enrichir category_gt: {e}")
            _get_idx_to_category._cache = None
    return _get_idx_to_category._cache


def enrich_details_with_categories(details: List[dict], model_name: str = "") -> List[dict]:
    """Remplit category_gt vide à partir du benchmark 3DSRBench (idx → category)."""
    need_enrich = any(not d.get("category_gt", "").strip() for d in details)
    if not need_enrich:
        return details
    idx2cat = _get_idx_to_category()
    if not idx2cat:
        return details
    n = 0
    for d in details:
        if not d.get("category_gt", "").strip():
            idx = d.get("idx", -1)
            if idx in idx2cat and idx2cat[idx]:
                d["category_gt"] = idx2cat[idx]
                n += 1
    if n:
        print(f"  [i] {model_name}: {n} category_gt complétés depuis le benchmark")
    return details


def compute_per_category(details: List[dict]) -> dict:
    """Compute accuracy per category. Returns {category: {correct, total, accuracy}}."""
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in details:
        cat = d.get("category_gt", "").strip()
        if not cat:
            continue
        gt = d.get("gt", "")
        pred = d.get("pred", "")
        by_cat[cat]["total"] += 1
        if pred and pred == gt:
            by_cat[cat]["correct"] += 1

    result = {}
    for cat in CATEGORIES:
        if cat in by_cat:
            n = by_cat[cat]["total"]
            c = by_cat[cat]["correct"]
            result[cat] = {"correct": c, "total": n, "accuracy": c / n if n else 0.0}
    # Catégories non standard (unknown, etc.)
    for cat, v in by_cat.items():
        if cat not in result:
            n = v["total"]
            c = v["correct"]
            result[cat] = {"correct": c, "total": n, "accuracy": c / n if n else 0.0}
    return result


def load_runs_from_file(runs_file: Path, base_dir: Path) -> List[tuple]:
    """Load runs from JSON: [{"name": "...", "path": "..."}]. Paths relative to base_dir.
    Returns (name, details_path). Si path = A/B, essaie aussi A/ (flat vs nested)."""
    with open(runs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = []
    base_dir = Path(base_dir).resolve()
    for r in data.get("runs", []):
        name = r.get("name", "")
        path_str = r.get("path", "")
        if not name or not path_str:
            continue
        full_path = (base_dir / path_str).resolve()
        details_path = full_path / "details.jsonl"
        parent_path = full_path.parent / "details.jsonl" if full_path.parent != base_dir else None
        # Priorité: path donné, puis parent (flat)
        candidates = [details_path]
        if parent_path and parent_path != details_path and parent_path.exists():
            candidates.append(parent_path)
        if not details_path.exists() and not (parent_path and parent_path.exists()):
            print(f"[!] Ignoré (details.jsonl absent): {path_str}")
            continue
        runs.append((name, candidates))
    return runs


def find_run_dirs(base: Path, mode: str) -> List[tuple[str, Path]]:
    """
    Find all (model_name, details_path) under base.
    Returns list of (model_name, path_to_details.jsonl).
    """
    runs = []
    base = Path(base).resolve()

    if mode == "api":
        # API: base = .../api_models/20260217_xxx or full_dataset
        # Subdirs: claude_sonnet_4_5_with_prompt, gpt4o_without_prompt, etc.
        for sub in sorted(base.iterdir()):
            if sub.is_dir():
                details_path = sub / "details.jsonl"
                if details_path.exists():
                    runs.append((sub.name, details_path))
    elif mode == "gpu":
        # GPU: base = .../3dsrbench
        # Subdirs: qwen3_4b/, llava4d/, sa2va/
        # Each has: full_dataset_with_prompt/, full_dataset_without_prompt/
        for model_dir in ["qwen3_4b", "llava4d", "sa2va"]:
            model_path = base / model_dir
            if not model_path.is_dir():
                continue
            for run_sub in model_path.iterdir():
                if run_sub.is_dir():
                    details_path = run_sub / "details.jsonl"
                    if details_path.exists():
                        name = f"{model_dir}_{run_sub.name}"
                        runs.append((name, details_path))
    else:
        # auto: recursive search for details.jsonl
        for details_path in base.rglob("details.jsonl"):
            rel = details_path.relative_to(base)
            # model name from path, e.g. api_models/xxx/claude_with_prompt or qwen3_4b/full_dataset_with_prompt
            parts = list(rel.parts[:-1])  # exclude details.jsonl
            name = "_".join(parts) if parts else details_path.parent.name
            runs.append((name, details_path))

    return runs


def main():
    parser = argparse.ArgumentParser(description="3DSRBench — Performance par catégorie")
    parser.add_argument(
        "--dir",
        default="results/runs/3dsrbench",
        help="Répertoire racine (api_models/xxx, ou 3dsrbench pour GPU)",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "gpu", "auto"],
        default="auto",
        help="api: sous-dirs directs avec details.jsonl | gpu: qwen3_4b/, llava4d/, sa2va/ | auto: recherche récursive",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Fichier de sortie (défaut: category_performance.csv dans --dir)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Répertoire de sortie (défaut: --dir)",
    )
    parser.add_argument(
        "--runs_file",
        default=None,
        help="JSON avec 6 runs exacts (name, path). Paths relatifs à --dir. Ex: scripts/evals/3dsrbench/runs_gpu_6.json",
    )
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.exists():
        abs_base = base.resolve()
        print(f"[ERREUR] --dir n'existe pas: {base}")
        if not base.is_absolute():
            print(f"  (absolu: {abs_base})")
        print("  Exécutez depuis la racine du projet (cd ~/CY/Spatial_MAS)")
        print("  Exemples:")
        print("    API:  --dir results/runs/3dsrbench/api_models/20260217_023920 --mode api")
        print("    API:  --dir results/runs/3dsrbench/api_models/full_dataset --mode api")
        print("    GPU:  --dir results/runs/3dsrbench --mode gpu")
        return 1

    # Pour API, si base se termine par api_models, on utilise le parent
    if args.mode == "api" and "api_models" in str(base):
        # base = results/runs/3dsrbench/api_models/20260217_xxx
        run_dir = base
    elif args.mode == "api":
        # base = results/runs/3dsrbench → chercher api_models/xxx
        api_base = base / "api_models"
        if api_base.exists():
            subdirs = sorted(api_base.iterdir(), key=lambda p: p.name, reverse=True)
            run_dir = subdirs[0] if subdirs else base
        else:
            run_dir = base
    else:
        run_dir = base

    if args.runs_file:
        runs_file = Path(args.runs_file)
        if not runs_file.exists():
            root = Path(__file__).resolve().parents[2]
            runs_file = root / args.runs_file
        if not runs_file.exists():
            print(f"[ERREUR] --runs_file introuvable: {args.runs_file}")
            return 1
        raw_runs = load_runs_from_file(runs_file, base)
        # Résoudre les candidats (path vs parent): prendre celui avec le plus de records
        runs = []
        for name, candidates in raw_runs:
            paths = [c for c in (candidates if isinstance(candidates, list) else [candidates]) if c.exists()]
            if not paths:
                continue
            best_path = max(paths, key=lambda p: sum(1 for d in load_details(p) if d.get("category_gt", "").strip()))
            runs.append((name, best_path))
    else:
        runs = find_run_dirs(run_dir, args.mode)
    if not runs:
        print(f"[ERREUR] Aucun details.jsonl trouvé sous {run_dir}")
        print("  Vérifiez --dir et --mode")
        return 1

    print(f"Trouvé {len(runs)} runs:")
    for name, details_path in runs:
        n = sum(1 for d in load_details(details_path) if d.get("category_gt", "").strip())
        print(f"  - {name} ({n} samples)")

    # Compute per-category for each model
    all_results = {}
    for model_name, details_path in runs:
        details = load_details(details_path)
        details = enrich_details_with_categories(details, model_name)
        per_cat = compute_per_category(details)
        total_with_cat = sum(v["total"] for v in per_cat.values())
        if total_with_cat == 0:
            print(f"[!] {model_name}: 0 samples avec category_gt dans {details_path}")
            print(f"    Vérifiez que le run est terminé et que details.jsonl contient des prédictions.")
        all_results[model_name] = per_cat

    # Build output
    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    out_csv = Path(args.output) if args.output else out_dir / "category_performance.csv"
    if args.output and str(out_csv).endswith(".csv"):
        out_json = out_csv.with_suffix(".json")
    else:
        out_json = out_dir / "category_performance.json"

    # CSV: rows = (model, category, accuracy, correct, total)
    rows = []
    for model_name, per_cat in all_results.items():
        for cat in CATEGORIES:
            if cat in per_cat:
                v = per_cat[cat]
                rows.append((model_name, cat, v["accuracy"], v["correct"], v["total"]))
        for cat, v in per_cat.items():
            if cat not in CATEGORIES:
                rows.append((model_name, cat, v["accuracy"], v["correct"], v["total"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("model,category,accuracy,correct,total\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]},{r[4]}\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table (overall + per category)
    print("\n" + "=" * 70)
    print("Performance par catégorie")
    print("=" * 70)

    # Header
    models = list(all_results.keys())
    cats_ordered = [c for c in CATEGORIES if any(c in all_results[m] for m in models)]
    header = "category".ljust(35) + "".join(m[:18].ljust(20) for m in models)
    print(header)
    print("-" * len(header))

    for cat in cats_ordered:
        line = cat.ljust(35)
        for m in models:
            v = all_results[m].get(cat, {})
            acc = v.get("accuracy", 0.0)
            n = v.get("total", 0)
            line += f"{acc:.2f} ({n})".ljust(20)
        print(line)

    print("=" * 70)
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    return 0


if __name__ == "__main__":
    exit(main())
