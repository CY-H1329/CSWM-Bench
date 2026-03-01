#!/usr/bin/env python3
"""
Test a specialist VLM across all 3 roles × 2 benchmarks × N samples.

Supports: sa2va, llava4d, qwen3_4b, spatial_rgpt, spatial_reasoner

Note: Sa2VA may fail with "bitsandbytes CUDA Setup failed" on some servers.
      Use --model qwen3_4b or --model llava4d as fallback.

Usage (CLI):
    python test_specialist_all_roles.py --model sa2va
    python test_specialist_all_roles.py --model qwen3_4b --max_samples 10

Usage (Jupyter):
    from test_specialist_all_roles import run_specialist_all_roles_test
    results = run_specialist_all_roles_test(model_name="qwen3_4b")
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_specialist_direct_visual import run_specialist_test as run_direct_visual
from test_specialist_explicit_3d import run_specialist_test as run_explicit_3d
from test_specialist_scene_graph import run_specialist_test as run_scene_graph


def _get_runner(model_name: str, device: str):
    """Load runner for given model. Raises on failure."""
    if model_name == "qwen3_4b":
        from src2.models.qwen3 import Qwen3Runner
        return Qwen3Runner(device=device)
    elif model_name == "sa2va":
        from src2.models.sa2va import Sa2VARunner
        return Sa2VARunner(device=device)
    elif model_name == "llava4d":
        from src2.models.llava import LLaVARunner
        return LLaVARunner(
            model_id="llava-hf/llava-v1.6-mistral-7b-hf",
            device=device,
        )
    elif model_name == "spatial_rgpt":
        from src2.models.spatial_rgpt import SpatialRGPTRunner
        return SpatialRGPTRunner(device=device)
    elif model_name == "spatial_reasoner":
        from src2.models.spatial_reasoner import SpatialReasonerRunner
        return SpatialReasonerRunner(
            model_id="ccvl/SpatialReasoner",
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: qwen3_4b, sa2va, llava4d, spatial_rgpt, spatial_reasoner"
        )


def run_specialist_all_roles_test(
    model_name: str = "sa2va",
    max_samples: int = 10,
    seed: int = 42,
    show_failures: int = 0,
    max_new_tokens: int = 1024,
    prefetch_workers: int = 0,
):
    """
    Run given model on all 3 roles × 2 benchmarks with max_samples each.

    Returns dict: { "model": str, "results": { (role, benchmark): {...} } }
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        runner = _get_runner(model_name, device)
    except RuntimeError as e:
        if "bitsandbytes" in str(e) or "CUDA Setup" in str(e):
            print("\n" + "=" * 70)
            print("MODEL LOAD FAILED: bitsandbytes/CUDA compatibility issue")
            print("=" * 70)
            print("Sa2VA (and some models) require bitsandbytes, which may fail")
            print("on this server. Try one of:")
            print("  --model qwen3_4b   (recommended)")
            print("  --model llava4d")
            print("  --model spatial_reasoner")
            print()
            print("To fix Sa2VA: run 'python -m bitsandbytes' to diagnose,")
            print("or install bitsandbytes matching your CUDA version.")
            print("=" * 70)
        raise
    except Exception as e:
        print(f"\nModel load failed: {e}")
        raise

    results = {}

    # 1. direct_visual_heuristic
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"{model_name} — direct_visual_heuristic — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_direct_visual(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            model_name=model_name,
        )
        if r:
            results[("direct_visual_heuristic", bench)] = r

    # 2. explicit_3d_representation
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"{model_name} — explicit_3d_representation — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_explicit_3d(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            model_name=model_name,
        )
        if r:
            results[("explicit_3d_representation", bench)] = r

    # 3. scene_graph_construction
    for bench in ["cvbench", "3dsrbench"]:
        print(f"\n{'='*60}")
        print(f"{model_name} — scene_graph_construction — {bench.upper()} (n={max_samples})")
        print("=" * 60)
        r = run_scene_graph(
            runner,
            benchmark=bench,
            max_samples=max_samples,
            seed=seed,
            show_failures=show_failures,
            max_new_tokens=max_new_tokens,
            prefetch_workers=prefetch_workers,
            model_name=model_name,
        )
        if r:
            results[("scene_graph_construction", bench)] = r

    # --- Summary ---
    print("\n")
    print("=" * 70)
    print(f"{model_name.upper()} — ALL ROLES × 2 BENCHMARKS — SUMMARY")
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

    return {"model": model_name, "results": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test specialist VLM across all 3 roles × 2 benchmarks")
    parser.add_argument("--model", default="qwen3_4b", help="qwen3_4b, sa2va, llava4d, spatial_rgpt, spatial_reasoner")
    parser.add_argument("--max_samples", type=int, default=10, help="Samples per (role, benchmark)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_failures", type=int, default=0, help="Print first N wrong cases per run")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prefetch_workers", type=int, default=0, help="Prefetch threads for scene_graph (0=off)")
    args = parser.parse_args()

    run_specialist_all_roles_test(
        model_name=args.model,
        max_samples=args.max_samples,
        seed=args.seed,
        show_failures=args.show_failures,
        max_new_tokens=args.max_new_tokens,
        prefetch_workers=args.prefetch_workers,
    )
