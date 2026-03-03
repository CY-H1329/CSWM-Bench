#!/usr/bin/env python3
"""
SpatialTTO: train 300 samples → score map 고정 → frozen benchmark으로 inference.

지원: cvbench, 3dsrbench

1. Train: data/dataset/<train_subdir> 300 samples, SpatialTTO로 score 업데이트 (4 agents)
2. Score map 저장
3. Eval: frozen benchmark으로 inference (TTO 업데이트 없음)

Usage:
    # CV-Bench
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench
    # 3DSRBench
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark 3dsrbench
    # Inference only
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark 3dsrbench --inference_only
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src2.agents.mas_v2 import ALL_CATEGORIES, ROLES, ScoreMap, run_test, compute_accuracy
from src2.benchmarks.loaders import (
    load_benchmark,
    load_benchmark_from_dataset,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
)
from src2.agents.mas_v2 import run_step
from test_confidence_mas_v3_step4 import TrustScoreMapUpdaterStep4, build_runners_for_confidence


BENCHMARK_CONFIG = {
    "cvbench": {
        "train_subdir": "cvbench_train_300",
        "output_dir": "results/spatialtto_300_frozen_cvbench",
        "score_map_name": "score_map_after_300.json",
        "frozen_size": 400,
    },
    "3dsrbench": {
        "train_subdir": "3dsrbench_train_300",
        "output_dir": "results/spatialtto_300_frozen_3dsrbench",
        "score_map_name": "score_map_after_300.json",
        "frozen_size": 500,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="cvbench", choices=["cvbench", "3dsrbench"],
                        help="Benchmark: cvbench or 3dsrbench")
    parser.add_argument("--train_samples", type=int, default=300,
                        help="Train samples (default: 300)")
    parser.add_argument("--train_subdir", type=str, default=None,
                        help="Override data/dataset/<subdir> (default: from benchmark)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (default: from benchmark)")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--eval_max", type=int, default=None,
                        help="Limit eval samples (None = full frozen)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true",
                        help="Skip train phase; load saved TTO score map and run eval only")
    parser.add_argument("--score_map_path", type=str, default=None,
                        help="Path to saved score_map JSON")
    args = parser.parse_args()

    bm_cfg = BENCHMARK_CONFIG[args.benchmark]
    train_subdir = args.train_subdir or bm_cfg["train_subdir"]
    out_dir = Path(args.output_dir or bm_cfg["output_dir"])
    score_map_name = bm_cfg["score_map_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    score_map_path = Path(args.score_map_path) if args.score_map_path else out_dir / score_map_name
    _bundled = PROJECT_ROOT / "data" / "score_maps" / "score_map_50step.json"

    specialist_whitelist = None
    if args.inference_only:
        if not score_map_path.exists() and _bundled.exists():
            score_map_path = _bundled
            print(f"[Info] Using bundled score map: {score_map_path}")
        if not score_map_path.exists():
            raise FileNotFoundError(
                f"Score map not found: {score_map_path}. "
                f"Run without --inference_only first to train on {train_subdir}."
            )
        score_map = ScoreMap.load(str(score_map_path))
        specialist_whitelist = score_map.llms
        specialist_llms = score_map.llms
        print(f"[Load] Score map from {score_map_path} (specialists: {specialist_llms})")
        correct_train = None
        train_samples = 0

    # --- 1. Build runners ---
    head_gen, spec_gen, reason_gen = build_runners_for_confidence(
        specialist_device="cuda",
        specialist_whitelist=specialist_whitelist,
        use_vlm_reasoning=True,
    )

    if not args.inference_only:
        specialist_llms = ["qwen3_4b", "llava4d", "spaceom", "spatial_reasoner"]
        # --- 2. Train: dataset with SpatialTTO ---
        train_path = PROJECT_ROOT / "data" / "dataset" / train_subdir
        if train_path.exists():
            train_ds = load_benchmark_from_dataset(
                args.benchmark, train_subdir,
                project_root=PROJECT_ROOT,
                max_samples=args.train_samples,
                seed=args.seed,
            )
            print(f"[Train] Loaded {len(train_ds)} samples from data/dataset/{train_subdir}")
        else:
            train_ds = load_benchmark(args.benchmark, max_samples=args.train_samples, use_frozen=False, seed=args.seed)
            print(f"[Train] Loaded {len(train_ds)} samples from HuggingFace (use_frozen=False)")

        print("\n" + "=" * 70)
        print(f"PHASE 1: Train ({len(train_ds)} samples, SpatialTTO updates, 4 agents incl. SpaceOm)")
        print("=" * 70)

        score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, roles=ROLES, seed=args.seed)
        updater = TrustScoreMapUpdaterStep4(T=args.T, kappa=args.kappa, gamma=args.gamma)

        import random
        rng = random.Random(args.seed)
        indices = list(range(len(train_ds)))
        rng.shuffle(indices)
        samples = [train_ds[i] for i in indices]

        correct_train = 0
        for step, ex in enumerate(samples):
            image = get_benchmark_image(ex, args.benchmark)
            if image is None:
                continue
            query = get_benchmark_prompt(ex, args.benchmark)
            gt_raw = get_benchmark_answer(ex, args.benchmark)
            gt = (gt_raw or "").strip().upper()
            if not any(c in gt for c in "ABCD"):
                continue

            result = run_step(
                image=image,
                query=query,
                gt=gt,
                step=step,
                total_steps=len(samples),
                score_map=score_map,
                head_generate=head_gen,
                specialist_generate=spec_gen,
                reasoning_generate=reason_gen,
                updater=updater,
                update_scores=True,
                use_vlm_reasoning=True,
            )
            if result.get("correct"):
                correct_train += 1
            acc = 100 * correct_train / (step + 1)
            log_every = 10 if len(samples) > 100 else 1
            if (step + 1) % log_every == 0 or step + 1 == len(samples):
                print(f"  Step {step + 1}/{len(samples)} | acc: {acc:.1f}% | cat: {result.get('category')} | assign: {result.get('assignments')}")

        print(f"\n[Train] Accuracy: {correct_train}/{len(samples)} = {100*correct_train/len(samples):.1f}%")
        train_samples = len(samples)

        # Save score map
        score_map.save(str(score_map_path))
        print(f"[Save] Score map → {score_map_path}")

    # --- 4. Eval: frozen benchmark (no TTO updates) ---
    frozen_name = {"cvbench": "cvbench_400", "3dsrbench": "3dsrbench_500"}[args.benchmark]
    print("\n" + "=" * 70)
    print(f"PHASE 2: Eval (frozen {frozen_name}, no TTO updates)")
    print("=" * 70)

    eval_ds = load_benchmark(args.benchmark, max_samples=args.eval_max, use_frozen=True, seed=args.seed)
    print(f"[Eval] Loaded {len(eval_ds)} samples from frozen {frozen_name}")

    eval_results = run_test(
        dataset=eval_ds,
        benchmark=args.benchmark,
        score_map=score_map,
        head_generate=head_gen,
        specialist_generate=spec_gen,
        reasoning_generate=reason_gen,
        random_agents=False,
        use_vlm_reasoning=True,
        verbose=True,
        updater=None,
        update_scores=False,
    )

    metrics = compute_accuracy(eval_results)
    per_cat = metrics.get("per_category", {})
    per_cat_counts = metrics.get("per_category_counts", {})

    print("\n" + "=" * 70)
    print(f"FINAL EVAL (frozen {args.benchmark})")
    print("=" * 70)
    print(f"Overall: {metrics['correct']}/{metrics['total']} = {100*metrics['accuracy']:.1f}%")
    print("-" * 70)
    print(f"{'Category':<14} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}")
    print("-" * 70)
    for cat, acc in sorted(per_cat.items(), key=lambda x: -x[1]):
        cnt = per_cat_counts.get(cat, {})
        c, t = cnt.get("correct", 0), cnt.get("total", 0)
        print(f"{cat:<14} | {c:>7} | {t:>5} | {100*acc:>6.1f}%")
    print("=" * 70)

    # Save eval summary (JSON + 발표용 카테고리별 표)
    import json
    from datetime import datetime
    per_cat_detail = {
        cat: {"correct": per_cat_counts.get(cat, {}).get("correct", 0),
              "total": per_cat_counts.get(cat, {}).get("total", 0),
              "accuracy": per_cat.get(cat, 0.0)}
        for cat in per_cat
    }
    summary = {
        "benchmark": args.benchmark,
        "inference_only": args.inference_only,
        "train_samples": train_samples,
        "train_accuracy": (correct_train / train_samples) if train_samples else None,
        "eval_total": metrics["total"],
        "eval_correct": metrics["correct"],
        "eval_accuracy": metrics["accuracy"],
        "per_category": per_cat_detail,
        "T": args.T,
        "kappa": args.kappa,
        "gamma": args.gamma,
        "specialist_llms": specialist_llms,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[Save] Eval summary → {out_dir / 'eval_summary.json'}")

    # 발표용 Markdown 테이블 (카테고리별 성능)
    md_lines = [
        f"## {args.benchmark.upper()} (frozen) - Category별 성능",
        "",
        f"**Overall:** {metrics['correct']}/{metrics['total']} = {100*metrics['accuracy']:.1f}%",
        "",
        "| Category | Correct | Total | Accuracy |",
        "|----------|---------|-------|----------|",
    ]
    for cat, acc in sorted(per_cat.items(), key=lambda x: -x[1]):
        cnt = per_cat_counts.get(cat, {})
        c, t = cnt.get("correct", 0), cnt.get("total", 0)
        md_lines.append(f"| {cat} | {c} | {t} | {100*acc:.1f}% |")
    md_lines.append("")
    (out_dir / "eval_by_category.md").write_text("\n".join(md_lines))
    print(f"[Save] Category table → {out_dir / 'eval_by_category.md'}")


if __name__ == "__main__":
    main()
