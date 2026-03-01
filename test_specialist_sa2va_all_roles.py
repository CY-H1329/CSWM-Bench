#!/usr/bin/env python3
"""
Test Sa2VA across all 3 specialist roles × 2 benchmarks × N samples.

If Sa2VA fails (bitsandbytes/CUDA), use test_specialist_all_roles.py --model qwen3_4b instead.

Usage (CLI):
    python test_specialist_sa2va_all_roles.py

Usage (Jupyter):
    from test_specialist_sa2va_all_roles import run_sa2va_all_roles_test
    results = run_sa2va_all_roles_test()
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_specialist_all_roles import run_specialist_all_roles_test


def run_sa2va_all_roles_test(
    max_samples: int = 10,
    seed: int = 42,
    show_failures: int = 0,
    max_new_tokens: int = 1024,
    prefetch_workers: int = 0,
):
    """Run Sa2VA on all 3 roles × 2 benchmarks. Delegates to run_specialist_all_roles_test."""
    return run_specialist_all_roles_test(
        model_name="sa2va",
        max_samples=max_samples,
        seed=seed,
        show_failures=show_failures,
        max_new_tokens=max_new_tokens,
        prefetch_workers=prefetch_workers,
    )


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
