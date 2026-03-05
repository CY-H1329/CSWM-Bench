#!/usr/bin/env python3
"""
Workflow complet:
1. Optimization (TTO) sur 3DSRBench 50 samples
2. Fixe les combinaisons (score map → assignments)
3. Inference sur CV-Bench 1000 (stratifié par catégorie)

Usage:
    python run_3dsrbench_opt_then_cvbench_inference.py

    # Options
    python run_3dsrbench_opt_then_cvbench_inference.py --skip_opt  # si score map déjà dispo
    python run_3dsrbench_opt_then_cvbench_inference.py --opt_only  # stop après optimization
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

OPT_OUT_DIR = PROJECT_ROOT / "results" / "spatialtto_50_3dsrbench"
SCORE_MAP_PATH = OPT_OUT_DIR / "score_map_after_50.json"
STEP_SUMMARY = OPT_OUT_DIR / "step_logs" / "train" / "step_050_summary.txt"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip_opt", action="store_true", help="Skip optimization, use existing score map")
    p.add_argument("--opt_only", action="store_true", help="Stop after optimization")
    p.add_argument("--specialist_offload", action="store_true")
    args = p.parse_args()

    # 1. Optimization 3DSRBench 50
    if not args.skip_opt:
        print("\n" + "=" * 70)
        print("PHASE 1: Optimization 3DSRBench (50 samples)")
        print("=" * 70)
        # Créer 3dsrbench_train_50 si absent
        train_50 = PROJECT_ROOT / "data" / "dataset" / "3dsrbench_train_50"
        if not train_50.exists():
            print("[Prep] Creating 3dsrbench_train_50...")
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_train_datasets.py"),
                 "--benchmarks", "3dsrbench", "--n", "50"],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
        opt_cmd = [
            sys.executable, str(PROJECT_ROOT / "run_confidence_mas_step4_train_then_eval_frozen.py"),
            "--benchmark", "3dsrbench_50",
        ]
        if args.specialist_offload:
            opt_cmd.append("--specialist_offload")
        ret = subprocess.run(opt_cmd, cwd=str(PROJECT_ROOT))
        if ret.returncode != 0:
            print("[Error] Optimization failed.")
            sys.exit(ret.returncode)
        if args.opt_only:
            print("[Done] Optimization complete. Run without --opt_only for inference.")
            return

    # Vérifier score map ou step summary
    if not SCORE_MAP_PATH.exists() and not STEP_SUMMARY.exists():
        print(f"[Error] Score map or step summary not found. Run optimization first.")
        print(f"  Expected: {SCORE_MAP_PATH}")
        print(f"  Or: {STEP_SUMMARY}")
        sys.exit(1)

    # 2. Inference CV-Bench 1000 (stratifié)
    print("\n" + "=" * 70)
    print("PHASE 2: Inference CV-Bench 1000 (stratifié par catégorie)")
    print("=" * 70)

    if STEP_SUMMARY.exists():
        assign_arg, assign_val = "--summary", str(STEP_SUMMARY)
    elif SCORE_MAP_PATH.exists():
        assign_arg, assign_val = "--score_map_path", str(SCORE_MAP_PATH)
    else:
        sys.exit(1)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_inference_fixed_spatialtto.py"),
        assign_arg, assign_val,
        "--benchmark", "cvbench",
        "--max_samples", "1000",
        "--max_per_category", "250",  # 4 cats × 250 = 1000
        "--output_dir", str(OPT_OUT_DIR),
    ]
    if args.specialist_offload:
        cmd.append("--specialist_offload")

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print("\n[Done] Full workflow complete.")


if __name__ == "__main__":
    main()
