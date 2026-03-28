#!/usr/bin/env python3
"""
TTO Step 1, 2, 3, 4 성능 비교: 70 step 동안 각 업데이트 규칙으로 학습 후 곡선 비교.

동일한 샘플 순서로 step1, step2, step3, step4를 각각 70 step 실행하고,
cumulative accuracy를 JSON에 기록한 뒤 그래프로 비교합니다.

Usage:
  python scripts/run_tto_step_comparison.py --benchmark cvbench
  python scripts/run_tto_step_comparison.py --benchmark stvqa --steps 70 --output results/tto_comparison
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src2.agents.mas_v2 import ALL_CATEGORIES, ROLES, ScoreMap, run_step, SPECIALIST_LLMS_5, SPECIALIST_LLMS_3
from src2.benchmarks.loaders import (
    load_benchmark,
    load_benchmark_from_dataset,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_category,
)


def get_updater(step_num: int, T: float = 5.0, kappa: float = 0.5, gamma: float = 0.3):
    """Return TrustScoreMapUpdater for step 1, 2, 3, or 4."""
    if step_num == 1:
        from test_confidence_mas_v3_step1 import TrustScoreMapUpdaterStep1
        return TrustScoreMapUpdaterStep1(kappa=kappa)
    if step_num == 2:
        from test_confidence_mas_v2_step2 import TrustScoreMapUpdaterStep2
        return TrustScoreMapUpdaterStep2(T=T, kappa=kappa)
    if step_num == 3:
        from test_confidence_mas_v3_step3 import TrustScoreMapUpdaterStep3
        return TrustScoreMapUpdaterStep3(T=T, kappa=kappa, gamma=gamma)
    if step_num == 4:
        from test_confidence_mas_v3_step4 import TrustScoreMapUpdaterStep4
        return TrustScoreMapUpdaterStep4(T=T, kappa=kappa, gamma=gamma)
    raise ValueError(f"Unknown step: {step_num}")


def run_tto_for_step(
    step_num: int,
    samples: list,
    benchmark: str,
    head_gen,
    spec_gen,
    reason_gen,
    max_steps: int,
    T: float,
    kappa: float,
    gamma: float,
    specialist_llms: list,
    seed: int,
) -> list:
    """
    Run TTO with given step updater for max_steps. Returns list of cumulative accuracy at each step.
    """
    score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, roles=ROLES, seed=seed)
    updater = get_updater(step_num, T=T, kappa=kappa, gamma=gamma)

    accuracies = []
    correct = 0
    processed = 0

    for step, ex in enumerate(samples):
        if processed >= max_steps:
            break
        image = get_benchmark_image(ex, benchmark)
        if image is None:
            continue
        query = get_benchmark_prompt(ex, benchmark)
        gt_raw = get_benchmark_answer(ex, benchmark)
        gt = (gt_raw or "").strip().upper()
        if benchmark != "stvqa" and not any(c in gt for c in "ABCD"):
            continue

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=processed,
            total_steps=max_steps,
            score_map=score_map,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            updater=updater,
            update_scores=True,
            use_vlm_reasoning=True,
            verbose_markdown=False,
            save_step_dir=None,
        )

        processed += 1
        if result.get("correct"):
            correct += 1
        acc = 100.0 * correct / processed
        accuracies.append(acc)

        ok = "✓" if result.get("correct") else "✗"
        print(f"    Step {step_num} | {processed}/{max_steps} | {ok} | acc: {acc:.1f}%")

    return accuracies


def _smooth_curve(y, window: int = 7):
    """Smooth curve with moving average."""
    import numpy as np
    if len(y) < window:
        return np.array(y)
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(np.array(y, dtype=float), sigma=window / 2.5, mode="nearest")
    except ImportError:
        kernel = np.ones(window) / window
        return np.convolve(np.array(y, dtype=float), kernel, mode="same")


def plot_comparison(data: dict, output_path: Path, max_steps: int = 70, smooth_window: int = 7, exclude_step1: bool = True):
    """Plot step2, step3, step4 accuracy curves (step1 excluded by default)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = {
        "step1": "Step 1 (s += R)",
        "step2": "Step 2 (s += R̃, γ=1)",
        "step3": "Step 3 (s += γ·R̃, γ=0.1)",
        "step4": "Step 4 (Beta + EMA)",
    }
    colors = {"step1": "#2ecc71", "step2": "#3498db", "step3": "#9b59b6", "step4": "#e74c3c"}
    keys = [k for k in ["step1", "step2", "step3", "step4"] if k not in (["step1"] if exclude_step1 else [])]

    for key in keys:
        if key not in data or not data[key]:
            continue
        accs = np.array(data[key], dtype=float)
        accs_smooth = _smooth_curve(accs, window=smooth_window)
        steps = np.arange(1, len(accs_smooth) + 1)
        ax.plot(steps, accs_smooth, label=labels.get(key, key), color=colors[key], linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12)
    title = "TTO Step 2 vs 3 vs 4 — Cumulative Accuracy" if exclude_step1 else "TTO Step 1 vs 2 vs 3 vs 4 — Cumulative Accuracy"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(1, max_steps)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TTO Step 1,2,3,4 comparison (70 steps each)")
    parser.add_argument("--benchmark", type=str, default="cvbench", choices=["cvbench", "3dsrbench", "stvqa"])
    parser.add_argument("--steps", type=int, default=500, help="Steps per TTO variant")
    parser.add_argument("--output", type=str, default="results/tto_step_comparison", help="Output dir for JSON + plot")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_spaceom", action="store_true",
                        help="Include SpaceOm (6 agents). Default: 5 agents")
    parser.add_argument("--low_memory", action="store_true",
                        help="Use 3 agents only (qwen3_4b, llava4d, spatial_reasoner)")
    parser.add_argument("--specialist_offload", action="store_true")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Only plot from existing JSON (path to tto_step_accuracies.json)")
    parser.add_argument("--include-step1", action="store_true", help="Include Step 1 in plot (default: exclude)")
    parser.add_argument("--smooth", type=int, default=7, help="Smoothing window for curves (default: 7)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature. 0=greedy, >0=sampling (e.g. 0.7).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p when temperature>0.")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.plot_only:
        json_path = Path(args.plot_only)
        if not json_path.exists():
            print(f"Error: {json_path} not found")
            sys.exit(1)
        raw = json.loads(json_path.read_text())
        data = {k: v for k, v in raw.items() if k != "meta" and isinstance(v, list)}
        meta = raw.get("meta", {})
        steps = meta.get("steps", max(len(v) for v in data.values()) if data else 70)
        plot_path = json_path.parent / "tto_step_comparison.png"
        plot_comparison(data, plot_path, max_steps=steps, smooth_window=args.smooth, exclude_step1=not args.include_step1)
        print(f"[Done] Plot → {plot_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (prefer *_train_500 when steps=500)
    base = {"cvbench": "cvbench", "3dsrbench": "3dsrbench", "stvqa": "stvqa"}[args.benchmark]
    train_subdir = f"{base}_train_{args.steps}" if args.steps in (300, 500) else f"{base}_train_300"
    train_path = PROJECT_ROOT / "data" / "dataset" / train_subdir
    if not train_path.exists() and args.steps == 500:
        train_subdir = f"{base}_train_300"
        train_path = PROJECT_ROOT / "data" / "dataset" / train_subdir

    # Auto-create dataset if missing (cvbench/stvqa 500)
    if not train_path.exists() and args.steps == 500 and args.benchmark in ("cvbench", "stvqa"):
        print(f"[Prepare] Creating {base}_train_500 (500 samples, stratified)...")
        import subprocess
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_train_datasets.py"),
             "--benchmarks", args.benchmark, "--n", "500"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        train_subdir = f"{base}_train_500"
        train_path = PROJECT_ROOT / "data" / "dataset" / train_subdir
    n_cats = {"cvbench": 4, "3dsrbench": 12, "stvqa": 9}[args.benchmark]
    max_per_cat = max((args.steps + n_cats - 1) // n_cats, {"cvbench": 50, "3dsrbench": 2, "stvqa": 25}[args.benchmark])

    if train_path.exists():
        train_ds = load_benchmark_from_dataset(
            args.benchmark, train_subdir,
            project_root=PROJECT_ROOT,
            max_samples=max(300, args.steps * 3),
            max_per_category=max_per_cat,
            seed=args.seed,
        )
    else:
        train_ds = load_benchmark(
            args.benchmark, max_samples=max(300, args.steps * 3),
            max_per_category=max_per_cat,
            use_frozen=False, seed=args.seed,
        )

    rng = random.Random(args.seed)
    indices = list(range(len(train_ds)))
    rng.shuffle(indices)
    samples = []
    for i in indices:
        ex = train_ds[i]
        img = get_benchmark_image(ex, args.benchmark)
        if img is None:
            continue
        gt_raw = get_benchmark_answer(ex, args.benchmark)
        gt = (gt_raw or "").strip().upper()
        if args.benchmark != "stvqa" and not any(c in gt for c in "ABCD"):
            continue
        samples.append(ex)
        if len(samples) >= args.steps:
            break

    n_steps = min(args.steps, len(samples))
    if len(samples) < args.steps:
        print(f"[Warn] Only {len(samples)} valid samples (need {args.steps}). Will run {n_steps} steps.")
    print(f"[Data] Loaded {len(samples)} samples for {n_steps} steps (benchmark={args.benchmark})")

    # Build runners
    from test_confidence_mas_v3_step4 import build_runners_for_confidence

    if args.low_memory:
        specialist_llms = SPECIALIST_LLMS_3
    elif args.with_spaceom:
        specialist_llms = list(SPECIALIST_LLMS_5) + ["spaceom"]
    else:
        specialist_llms = list(SPECIALIST_LLMS_5)
    head_gen, spec_gen, reason_gen = build_runners_for_confidence(
        specialist_device="cuda",
        specialist_whitelist=specialist_llms,
        use_vlm_reasoning=True,
        specialist_offload_after_use=args.specialist_offload,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Run each TTO step
    data = {}
    for step_num in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"TTO Step {step_num} ({n_steps} steps)")
        print("=" * 60)
        accs = run_tto_for_step(
            step_num=step_num,
            samples=samples,
            benchmark=args.benchmark,
            head_gen=head_gen,
            spec_gen=spec_gen,
            reason_gen=reason_gen,
            max_steps=n_steps,
            T=args.T,
            kappa=args.kappa,
            gamma=args.gamma,
            specialist_llms=specialist_llms,
            seed=args.seed,
        )
        data[f"step{step_num}"] = accs
        print(f"  Final: {accs[-1]:.1f}%" if accs else "  (no data)")

    # Save JSON
    json_path = out_dir / "tto_step_accuracies.json"
    meta = {
        "benchmark": args.benchmark,
        "steps": n_steps,
        "T": args.T,
        "kappa": args.kappa,
        "gamma": args.gamma,
        "seed": args.seed,
    }
    out_json = {"meta": meta, **data}
    json_path.write_text(json.dumps(out_json, indent=2))
    print(f"\n[Save] JSON → {json_path}")

    # Plot
    plot_path = out_dir / "tto_step_comparison.png"
    plot_comparison(data, plot_path, max_steps=n_steps, smooth_window=args.smooth, exclude_step1=not args.include_step1)
    print(f"[Done] Results in {out_dir}")


if __name__ == "__main__":
    main()
