#!/usr/bin/env python3
"""
Test Sa2VA across all 3 specialist roles × 2 benchmarks × 10 samples.

Measures whether Sa2VA responds well as each specialist agent (direct_visual,
explicit_3d, scene_graph) on CV-Bench and 3DSRBench.

Usage (CLI):
    python test_specialist_sa2va_all_roles.py

Usage (Jupyter):
    from test_specialist_sa2va_all_roles import run_sa2va_all_roles_test
    results = run_sa2va_all_roles_test()
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_specialist_direct_visual import run_specialist_test as run_direct_visual
from test_specialist_explicit_3d import run_specialist_test as run_explicit_3d
from test_specialist_scene_graph import run_specialist_test as run_scene_graph


def run_sa2va_all_roles_test(
    max_samples: int = 10,
    seed: int = 42,
    show_failures: int = 0,
    max_new_tokens: int = 1024,
    prefetch_workers: int = 0,
):
    """
    Run Sa2VA on all 3 roles × 2 benchmarks with max_samples each.

    Returns dict: { (role, benchmark): { correct, total, accuracy, ... } }
    """
    import torch
    from src2.models.sa2va import Sa2VARunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = Sa2VARunner(device=device)

    results = {}

    # 1. direct_visual_heuristic
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"Sa2VA — direct_visual_heuristic — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_direct_visual(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            model_name="sa2va",
        )
        if r:
            results[("direct_visual_heuristic", bench)] = r

    # 2. explicit_3d_representation
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"Sa2VA — explicit_3d_representation — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_explicit_3d(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            model_name="sa2va",
        )
        if r:
            results[("explicit_3d_representation", bench)] = r

    # 3. scene_graph_construction
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"Sa2VA — scene_graph_construction — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_scene_graph(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            prefetch_workers=prefetch_workers,
            model_name="sa2va",
        )
        if r:
            results[("scene_graph_construction", bench)] = r

    # --- Summary ---
    print("\n")
    print("=" * 70)
    print("Sa2VA — ALL ROLES × 2 BENCHMARKS — SUMMARY")
    print("=" * 70)
    print(f"{'Role':<35} {'CV-Bench':>12} {'3DSRBench':>12}")
    print("-" * 70)

    for role in ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]:
        cv = results.get((role, "cvbench"), {})
        d3 = results.get((role, "3dsrbench"), {})
        cv_str = f"{cv.get('correct', 0)}/{cv.get('total', 0)} ({100*cv.get('accuracy', 0):.1f}%)" if cv else "-"
        d3_str = f"{d3.get('correct', 0)}/{d3.get('total', 0)} ({100*d3.get('accuracy', 0):.1f}%)" if d3 else "-"
        print(f"{role:<35} {cv_str:>12} {d3_str:>12}")

    print("=" * 70)

    return {"model": "sa2va", "results": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Sa2VA across all 3 roles × 2 benchmarks")
    parser.add_argument("--max_samples", type=int, default=10, help="Samples per (role, benchmark)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_failures", type=int, default=0, help="Print first N wrong cases per run")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prefetch_workers", type=int, default=0, help="Prefetch threads for scene_graph (0=off)")
    args = parser.parse_args()

    run_sa2va_all_roles_test(
        max_samples=args.max_samples,
        seed=args.seed,
        show_failures=args.show_failures,
        max_new_tokens=args.max_new_tokens,
        prefetch_workers=args.prefetch_workers,
    )
