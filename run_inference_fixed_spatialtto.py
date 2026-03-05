#!/usr/bin/env python3
"""
SpatialTTO Inference avec combinaison fixe (par tâche).

Charge les assignments depuis un step summary (step_050_summary.txt) ou un JSON,
et exécute l'inférence sur CV-Bench 500 sans mise à jour des scores.
Tout est figé : Head Agent + routing + combinaison agent×rôle par catégorie.

Usage:
    # Depuis step summary
    python run_inference_fixed_spatialtto.py --summary /Users/flaxinger/Downloads/step_050_summary.txt --benchmark cvbench --max_samples 500

    # Depuis JSON (généré par scripts/parse_step_summary.py)
    python run_inference_fixed_spatialtto.py --assignment_json /tmp/step_050_assignments.json --benchmark cvbench --max_samples 500
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src2.agents.mas_v2 import ALL_CATEGORIES, ROLES, run_step, run_test, compute_accuracy
from src2.agents.mas_v2.config import FINE_TO_UNIFIED
from src2.agents.mas_v2.score_map import ScoreMap
from src2.benchmarks.loaders import (
    load_benchmark,
    load_benchmark_from_dataset,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_category,
)
from test_confidence_mas_v3_step4 import build_runners_for_confidence


# ---------------------------------------------------------------------------
# FixedAssignmentScoreMap
# ---------------------------------------------------------------------------
class FixedAssignmentScoreMap(ScoreMap):
    """ScoreMap qui retourne une combinaison fixe par catégorie."""

    def __init__(
        self,
        assignments: Dict[str, List[Tuple[str, str]]],
        categories: List[str],
        llms: List[str],
    ):
        super().__init__(categories=categories, llms=llms, roles=ROLES)
        self._fixed: Dict[str, List[Tuple[str, str]]] = dict(assignments)
        self._fallback = list(self._fixed.values())[0] if self._fixed else []

    def select_agents(self, category: str, step: int) -> List[Tuple[str, str]]:
        lst = self._fixed.get(category) or self._fixed.get(
            category.lower().replace(" ", "_")
        )
        if lst:
            return [(r, a) for r, a in lst]
        return self._fallback


def load_assignments_from_summary(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse step summary via scripts/parse_step_summary.py."""
    from scripts.parse_step_summary import parse_step_summary
    raw = parse_step_summary(path)
    return {cat: [(r, a) for r, a in lst] for cat, lst in raw.items()}


def load_assignments_from_json(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Charge assignments depuis JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    a = data.get("assignments", data)
    return {
        cat: [(r, ag) for r, ag in lst]
        for cat, lst in a.items()
    }


def load_assignments_from_score_map(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Dérive assignments depuis score_map JSON (argmax par rôle, sans réutilisation)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    maps = data.get("maps", {})
    llms = data.get("llms", ["qwen3_4b", "llava4d", "spatial_reasoner"])
    roles = data.get("roles", ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"])
    assignments = {}
    for cat, cat_map in maps.items():
        used = set()
        lst = []
        for role in roles:
            role_scores = cat_map.get(role, {})
            best_agent, best_s = None, -1e9
            for agent in llms:
                if agent in used:
                    continue
                s = role_scores.get(agent, 0.5)
                if s > best_s:
                    best_s, best_agent = s, agent
            if best_agent:
                lst.append((role, best_agent))
                used.add(best_agent)
            elif llms:
                lst.append((role, llms[0]))
        assignments[cat] = lst
    return assignments


def main():
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", type=str, help="Path to step_XXX_summary.txt")
    g.add_argument("--assignment_json", type=str, help="Path to assignments JSON")
    g.add_argument("--score_map_path", type=str, help="Path to score_map JSON (derive assignments)")
    p.add_argument("--benchmark", type=str, default="cvbench")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_per_category", type=int, default=None,
                   help="Stratified: max samples per category (e.g. 250 for 1000 total on 4 CV-Bench cats)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--specialist_offload", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true", help="Full verbose (cat, assign, scores). Default: minimal (step + O/X)")
    args = p.parse_args()

    if args.summary:
        assignments = load_assignments_from_summary(args.summary)
        print(f"[Load] Assignments from step summary: {args.summary}")
    elif args.assignment_json:
        assignments = load_assignments_from_json(args.assignment_json)
        print(f"[Load] Assignments from JSON: {args.assignment_json}")
    else:
        assignments = load_assignments_from_score_map(args.score_map_path)
        print(f"[Load] Assignments from score map: {args.score_map_path}")

    llms = []
    for lst in assignments.values():
        for _, a in lst:
            if a not in llms:
                llms.append(a)
    if not llms:
        llms = ["qwen3_4b", "llava4d", "spatial_reasoner"]
    print(f"[Info] Specialists: {llms}")

    score_map = FixedAssignmentScoreMap(
        assignments=assignments,
        categories=ALL_CATEGORIES,
        llms=llms,
    )

    # Charger CV-Bench (stratifié si max_per_category)
    train_n = PROJECT_ROOT / "data" / "dataset" / f"cvbench_train_{args.max_samples}"
    if args.max_per_category is not None:
        # Stratifié: max_per_category par catégorie (ex: 250 × 4 = 1000)
        dataset = load_benchmark(
            "cvbench",
            max_per_category=args.max_per_category,
            use_frozen=False,
            seed=args.seed,
        )
        print(f"[Eval] Loaded {len(dataset)} from CV-Bench (stratified, {args.max_per_category}/cat)")
    elif train_n.exists():
        dataset = load_benchmark_from_dataset(
            "cvbench", f"cvbench_train_{args.max_samples}",
            project_root=PROJECT_ROOT,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"[Eval] Loaded {len(dataset)} from data/dataset/cvbench_train_{args.max_samples}")
    else:
        import subprocess
        prep = PROJECT_ROOT / "scripts" / "prepare_train_datasets.py"
        if prep.exists() and args.max_samples in (300, 500):
            print(f"[Info] cvbench_train_{args.max_samples} absent. Création...")
            subprocess.run(
                [sys.executable, str(prep), "--benchmarks", "cvbench", "--n", str(args.max_samples)],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            dataset = load_benchmark_from_dataset(
                "cvbench", f"cvbench_train_{args.max_samples}",
                project_root=PROJECT_ROOT,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            print(f"[Eval] Loaded {len(dataset)} from data/dataset/cvbench_train_{args.max_samples}")
        else:
            dataset = load_benchmark(
                "cvbench",
                max_samples=args.max_samples,
                max_per_category=args.max_per_category,
                use_frozen=False,
                seed=args.seed,
            )
            print(f"[Eval] Loaded {len(dataset)} from CV-Bench (HuggingFace)")

    head_gen, spec_gen, reason_gen = build_runners_for_confidence(
        specialist_device="cuda",
        specialist_whitelist=llms,
        use_vlm_reasoning=True,
        specialist_offload_after_use=args.specialist_offload,
    )

    print("\n" + "=" * 70)
    print(f"SPATIALTTO INFERENCE — CV-Bench {len(dataset)} samples (fixed assignment)")
    print("=" * 70)

    results = run_test(
        dataset=dataset,
        benchmark=args.benchmark,
        score_map=score_map,
        head_generate=head_gen,
        specialist_generate=spec_gen,
        reasoning_generate=reason_gen,
        random_agents=False,
        use_vlm_reasoning=True,
        verbose=args.verbose,
        verbose_minimal=not args.verbose,
        verbose_markdown=False,
        updater=None,
        update_scores=False,
    )

    metrics = compute_accuracy(results)
    per_cat = metrics.get("per_category", {})
    per_cat_counts = metrics.get("per_category_counts", {})

    print("\n" + "=" * 70)
    print("FINAL — CV-Bench (fixed assignment)")
    print("=" * 70)
    print(f"Overall: {metrics['correct']}/{metrics['total']} = {100*metrics['accuracy']:.1f}%")
    print("-" * 70)
    print(f"{'Category':<18} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}")
    print("-" * 70)
    for cat, acc in sorted(per_cat.items(), key=lambda x: -x[1]):
        cnt = per_cat_counts.get(cat, {})
        c, t = cnt.get("correct", 0), cnt.get("total", 0)
        print(f"{cat:<18} | {c:>7} | {t:>5} | {acc*100:>6.1f}%")
    print("=" * 70)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / "inference_fixed_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "correct": metrics["correct"],
                "total": metrics["total"],
                "accuracy": metrics["accuracy"],
                "per_category": per_cat,
                "per_category_counts": per_cat_counts,
            }, f, indent=2)
        print(f"\n[Save] Results → {out_file}")


if __name__ == "__main__":
    main()
