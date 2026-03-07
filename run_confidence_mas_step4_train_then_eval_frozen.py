#!/usr/bin/env python3
"""
SpatialTTO: train → score map 고정 → frozen benchmark으로 inference.

지원: cvbench (200), 3dsrbench (20), stvqa (200) — 기본 3 agents (SpaceOm 제외)

1. Train: data/dataset/<train_subdir> (GitHub 데이터), SpatialTTO로 confidence score 업데이트
2. Score map 저장
3. Eval: frozen benchmark으로 inference (TTO 업데이트 없음)

Usage:
    # CV-Bench: 200 samples
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench
    # CV-Bench 150-step optimization (combination + score from 150 steps)
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench_150 --eval
    # CV-Bench full: ~2638 samples (HuggingFace test split)
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench_full --eval
    # 3DSRBench: 20 samples
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark 3dsrbench
    # STVQA-7K: 200 samples (data/dataset/stvqa_train_300, run: python scripts/prepare_train_datasets.py --datasets stvqa)
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark stvqa
    # Inference only
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark 3dsrbench --inference_only
    # Train only (TTO, skip eval) — default
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench
    # Train + Eval
    python run_confidence_mas_step4_train_then_eval_frozen.py --benchmark cvbench --eval
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src2.agents.mas_v2 import ALL_CATEGORIES, ROLES, ScoreMap, run_test, compute_accuracy
try:
    from src2.agents.mas_v2 import SPECIALIST_LLMS_5, SPECIALIST_LLMS_3
except ImportError:
    try:
        from src2.agents.mas_v2.config import SPECIALIST_LLMS_5, SPECIALIST_LLMS_3
    except ImportError:
        # Fallback when config.py is outdated (e.g. H100)
        SPECIALIST_LLMS_5 = ["qwen3_4b", "sa2va", "llava4d", "spatial_rgpt", "spatial_reasoner"]
        SPECIALIST_LLMS_3 = ["qwen3_4b", "llava4d", "spatial_reasoner"]
from src2.benchmarks.loaders import (
    FROZEN_PATHS,
    load_benchmark,
    load_benchmark_from_dataset,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_category,
)
from src2.agents.mas_v2 import run_step
from test_confidence_mas_v3_step4 import TrustScoreMapUpdaterStep4, build_runners_for_confidence


BENCHMARK_CONFIG = {
    "cvbench": {
        "train_subdir": "cvbench_train_300",
        "train_samples": 200,
        "max_per_category": 50,  # 4 cats × 50 = 200
        "output_dir": "results/spatialtto_200_frozen_cvbench",
        "score_map_name": "score_map_after_200.json",
        "frozen_size": 400,
    },
    "cvbench_150": {
        "train_subdir": "cvbench_train_300",
        "train_samples": 150,
        "max_per_category": 38,  # 4 cats × 38 ≈ 150
        "output_dir": "results/spatialtto_150_frozen_cvbench",
        "score_map_name": "score_map_after_150.json",
        "frozen_size": 400,
    },
    "cvbench_full": {
        "train_subdir": None,  # Load full from HuggingFace (no local dataset)
        "train_samples": None,  # Full = all ~2638
        "max_per_category": None,
        "output_dir": "results/spatialtto_full_cvbench",
        "score_map_name": "score_map_after_full.json",
        "frozen_size": 2638,
        "use_frozen_for_eval": False,  # Eval on full HF, not cvbench_400
    },
    "3dsrbench": {
        "train_subdir": "3dsrbench_train_300",
        "train_samples": 20,
        "max_per_category": 2,  # 12 cats × 2 = 24, cap at 20
        "output_dir": "results/spatialtto_20_frozen_3dsrbench",
        "score_map_name": "score_map_after_20.json",
        "frozen_size": 500,
    },
    "3dsrbench_50": {
        "train_subdir": "3dsrbench_train_50",
        "train_samples": 50,
        "max_per_category": 5,  # 12 cats × 5 = 60, cap at 50
        "output_dir": "results/spatialtto_50_3dsrbench",
        "score_map_name": "score_map_after_50.json",
        "frozen_size": 500,
    },
    "stvqa": {
        "train_subdir": "stvqa_train_300",
        "train_samples": 200,
        "max_per_category": 25,  # 9 cats × 25 = 225, cap at 200
        "output_dir": "results/spatialtto_200_frozen_stvqa",
        "score_map_name": "score_map_after_200.json",
        "frozen_size": 692,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="cvbench",
                        choices=["cvbench", "cvbench_150", "cvbench_full", "3dsrbench", "3dsrbench_50", "stvqa"],
                        help="Benchmark: cvbench, cvbench_150 (150-step opt), cvbench_full, 3dsrbench, 3dsrbench_50, stvqa")
    parser.add_argument("--train_samples", type=int, default=None,
                        help="Train samples (default: from benchmark config)")
    parser.add_argument("--train_subdir", type=str, default=None,
                        help="Override data/dataset/<subdir> (default: from benchmark)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir (default: from benchmark)")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--eval_max", type=int, default=None,
                        help="Limit eval samples (None = full frozen)")
    parser.add_argument("--eval_full", action="store_true",
                        help="Eval on full CV-Bench (~2638) from HuggingFace instead of cvbench_400")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true",
                        help="Skip train phase; load saved TTO score map and run eval only")
    parser.add_argument("--eval", action="store_true",
                        help="Run eval phase after train (default: train only)")
    parser.add_argument("--score_map_path", type=str, default=None,
                        help="Path to saved score_map JSON")
    parser.add_argument("--with_spaceom", action="store_true",
                        help="Include SpaceOm (6 agents). Default: 5 agents (qwen3_4b, sa2va, llava4d, spatial_rgpt, spatial_reasoner)")
    parser.add_argument("--low_memory", action="store_true",
                        help="Use 3 agents only (qwen3_4b, llava4d, spatial_reasoner) for OOM / quick test")
    parser.add_argument("--no_spatial_rgpt", action="store_true",
                        help="Exclude SpatialRGPT (4 agents: qwen3_4b, sa2va, llava4d, spatial_reasoner)")
    parser.add_argument("--specialist_offload", action="store_true",
                        help="Offload specialists to CPU after use (saves GPU memory, slower)")
    parser.add_argument("--verbose_markdown", action="store_true",
                        help="Print markdown summary per step (question, category, scores, agents, updated table)")
    parser.add_argument("--save_step_text", type=str, default=None,
                        help="Save step data to text files in this dir: routing, per-agent CoT, final")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N steps (partial run, e.g. 200). Saves score_map + summary.")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save score_map every N steps (default 50). Use 0 to disable.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature. 0=greedy, >0=sampling (e.g. 0.7).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p when temperature>0.")
    args = parser.parse_args()

    # 3dsrbench_50, cvbench_full, cvbench_150 use same loader as base benchmark
    benchmark_load = "3dsrbench" if args.benchmark == "3dsrbench_50" else (
        "cvbench" if args.benchmark in ("cvbench_full", "cvbench_150") else args.benchmark
    )
    bm_cfg = BENCHMARK_CONFIG[args.benchmark]
    train_subdir = args.train_subdir if args.train_subdir is not None else bm_cfg.get("train_subdir")
    train_samples = args.train_samples if args.train_samples is not None else bm_cfg.get("train_samples")
    max_per_category = bm_cfg.get("max_per_category")
    out_dir = Path(args.output_dir or bm_cfg["output_dir"])
    score_map_name = bm_cfg["score_map_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_step_dir = None
    if args.save_step_text:
        save_step_dir = Path(args.save_step_text)
        save_step_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Save] Step text files → {save_step_dir}/{{train,eval}}/")
    elif not args.inference_only:
        # Optimization (train): 기본으로 모든 로그 저장
        save_step_dir = out_dir / "step_logs"
        save_step_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Save] Step text files (train) → {save_step_dir}/train/")
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
        specialist_llms = score_map.llms
        if args.no_spatial_rgpt:
            specialist_whitelist = [a for a in specialist_llms if a != "spatial_rgpt"]
            score_map.llms = specialist_whitelist  # exclude from selection
        else:
            specialist_whitelist = specialist_llms
        print(f"[Load] Score map from {score_map_path} (specialists: {specialist_llms})")
        correct_train = None
        train_samples = 0

    if not args.inference_only:
        if args.low_memory:
            specialist_llms = SPECIALIST_LLMS_3
        elif args.with_spaceom:
            specialist_llms = ["qwen3_4b", "sa2va", "llava4d", "spatial_rgpt", "spaceom", "spatial_reasoner"]
        else:
            specialist_llms = SPECIALIST_LLMS_5
        if args.no_spatial_rgpt:
            specialist_llms = [a for a in specialist_llms if a != "spatial_rgpt"]
        specialist_whitelist = specialist_llms

    # --- 1. Build runners ---
    def _build():
        kw = dict(
            specialist_device="cuda",
            specialist_whitelist=specialist_whitelist,
            use_vlm_reasoning=True,
        )
        try:
            return build_runners_for_confidence(
                **kw, specialist_offload_after_use=args.specialist_offload,
                temperature=args.temperature, top_p=args.top_p,
            )
        except TypeError:
            pass
        try:
            return build_runners_for_confidence(**kw, specialist_offload_after_use=args.specialist_offload)
        except TypeError:
            return build_runners_for_confidence(**kw)

    head_gen, spec_gen, reason_gen = _build()

    if not args.inference_only:
        # --- 2. Train: dataset with SpatialTTO ---
        train_path = (PROJECT_ROOT / "data" / "dataset" / train_subdir) if train_subdir else None
        if train_path and train_path.exists():
            train_ds = load_benchmark_from_dataset(
                benchmark_load, train_subdir,
                project_root=PROJECT_ROOT,
                max_samples=train_samples,
                max_per_category=max_per_category,
                seed=args.seed,
            )
            print(f"[Train] Loaded {len(train_ds)} samples from data/dataset/{train_subdir} (max_per_cat={max_per_category})")
        else:
            # cvbench_full or no local dataset: load from HuggingFace (full when max_samples=max_per_category=None)
            train_ds = load_benchmark(
                benchmark_load, max_samples=train_samples,
                max_per_category=max_per_category,
                use_frozen=False, seed=args.seed,
            )
            src = "full HuggingFace" if (train_samples is None and max_per_category is None) else "HuggingFace"
            print(f"[Train] Loaded {len(train_ds)} samples from {src} (use_frozen=False)")

        print("\n" + "=" * 70)
        n_agents = len(specialist_llms)
        print(f"PHASE 1: Train ({len(train_ds)} samples, SpatialTTO updates, {n_agents} agents)")
        print("=" * 70)

        score_map = ScoreMap(categories=ALL_CATEGORIES, llms=specialist_llms, roles=ROLES, seed=args.seed)
        updater = TrustScoreMapUpdaterStep4(T=args.T, kappa=args.kappa, gamma=args.gamma)

        import random
        from collections import defaultdict
        rng = random.Random(args.seed)
        indices = list(range(len(train_ds)))
        rng.shuffle(indices)
        samples = [train_ds[i] for i in indices]

        correct_train = 0
        per_cat_train = defaultdict(lambda: {"correct": 0, "total": 0})
        n_samples = len(samples)
        if args.max_steps is not None:
            n_samples = min(n_samples, args.max_steps)
            print(f"[Info] Limiting to {n_samples} steps (--max_steps {args.max_steps})")
        try:
            for step, ex in enumerate(samples):
                if args.max_steps is not None and step >= args.max_steps:
                    break
                image = get_benchmark_image(ex, benchmark_load)
                if image is None:
                    continue
                query = get_benchmark_prompt(ex, benchmark_load)
                gt_raw = get_benchmark_answer(ex, benchmark_load)
                gt = (gt_raw or "").strip().upper()
                # Skip only for multiple-choice benchmarks when answer has no A/B/C/D
                if args.benchmark != "stvqa" and not any(c in gt for c in "ABCD"):
                    continue

                _step_dir = (save_step_dir / "train") if save_step_dir else None
                _train_verbose_md = args.verbose_markdown or bool(not args.inference_only)  # Optimization: 기본으로 모든 로그
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
                    verbose_markdown=_train_verbose_md,
                    save_step_dir=_step_dir,
                )
                if _train_verbose_md and result.get("verbose_markdown"):
                    print(result["verbose_markdown"])
                if result.get("correct"):
                    correct_train += 1
                cat = get_benchmark_category(ex, benchmark_load) or result.get("category") or "unknown"
                per_cat_train[cat]["total"] += 1
                if result.get("correct"):
                    per_cat_train[cat]["correct"] += 1
                acc = 100 * correct_train / (step + 1)
                pred = (result.get("final_answer") or "").strip()[:20]
                gt_s = (result.get("gt") or "").strip()[:20]
                ok = "✓" if result.get("correct") else "✗"
                print(f"  Step {step + 1}/{len(samples)} | {ok} | acc: {acc:.1f}% | cat: {result.get('category')} | assign: {result.get('assignments')} | pred: {pred} | gt: {gt_s}")
                # 카테고리별 성능 실시간 출력
                cat_lines = []
                for c in sorted(per_cat_train.keys()):
                    t = per_cat_train[c]["total"]
                    cr = per_cat_train[c]["correct"]
                    pct = 100 * cr / t if t else 0
                    cat_lines.append(f"{c}:{cr}/{t}({pct:.0f}%)")
                print("    " + " | ".join(cat_lines))

                # Checkpoint: save score_map every N steps
                if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                    score_map.save(str(score_map_path))
                    print(f"  [Checkpoint] Step {step + 1} → {score_map_path}")

        except KeyboardInterrupt:
            print("\n[Interrupted] Saving partial results...")
            score_map.save(str(score_map_path))
            train_samples = sum(per_cat_train[c]["total"] for c in per_cat_train)
            correct_train = sum(per_cat_train[c]["correct"] for c in per_cat_train)
            n_samples = train_samples if train_samples else step + 1
            print(f"  Saved partial: {correct_train}/{n_samples} steps → {score_map_path}")

        print(f"\n[Train] Accuracy: {correct_train}/{n_samples} = {100*correct_train/n_samples:.1f}%")
        print("-" * 50)
        print(f"{'Category':<14} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}")
        print("-" * 50)
        for cat in sorted(per_cat_train.keys()):
            c, t = per_cat_train[cat]["correct"], per_cat_train[cat]["total"]
            acc = 100 * c / t if t else 0
            print(f"{cat:<14} | {c:>7} | {t:>5} | {acc:>6.1f}%")
        print("-" * 50)
        train_samples = n_samples

        # Save score map
        score_map.save(str(score_map_path))
        print(f"[Save] Score map → {score_map_path}")

    # --- 4. Eval: frozen benchmark (no TTO updates) ---
    if args.eval:
        use_frozen_eval = False if args.eval_full else bm_cfg.get("use_frozen_for_eval", True)
        eval_max = None if args.eval_full else args.eval_max
        frozen_name = FROZEN_PATHS.get(benchmark_load, "cvbench_400") if use_frozen_eval else "HuggingFace (full ~2638)"
        print("\n" + "=" * 70)
        print(f"PHASE 2: Eval ({frozen_name}, no TTO updates)")
        print("=" * 70)

        eval_ds = load_benchmark(
            benchmark_load, max_samples=eval_max,
            use_frozen=use_frozen_eval, seed=args.seed,
        )
        print(f"[Eval] Loaded {len(eval_ds)} samples from {frozen_name}")

        _eval_step_dir = (save_step_dir / "eval") if save_step_dir else None
        eval_results = run_test(
            dataset=eval_ds,
            benchmark=benchmark_load,
            score_map=score_map,
            head_generate=head_gen,
            specialist_generate=spec_gen,
            reasoning_generate=reason_gen,
            random_agents=False,
            use_vlm_reasoning=True,
            verbose=True,
            verbose_markdown=args.verbose_markdown,
            updater=None,
            update_scores=False,
            save_step_dir=_eval_step_dir,
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
    else:
        # train only (no eval): save train summary + per-category accuracy
        if not args.inference_only:
            import json
            from datetime import datetime
            per_cat_detail = {}
            for cat, cnt in per_cat_train.items():
                t = cnt["total"]
                c = cnt["correct"]
                per_cat_detail[cat] = {
                    "correct": c,
                    "total": t,
                    "accuracy": (c / t) if t else 0.0,
                }
            summary = {
                "benchmark": args.benchmark,
                "train_only": True,
                "train_samples": train_samples,
                "train_accuracy": (correct_train / train_samples) if train_samples else None,
                "per_category": per_cat_detail,
                "T": args.T,
                "kappa": args.kappa,
                "gamma": args.gamma,
                "specialist_llms": specialist_llms,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"\n[Save] Train summary → {out_dir / 'train_summary.json'}")
            # 카테고리별 accuracy 로그
            md_lines = [
                "## Train — Category별 Accuracy",
                "",
                "**Overall:** {}/{} = {:.1f}%".format(
                    correct_train, train_samples,
                    100 * correct_train / train_samples if train_samples else 0,
                ),
                "",
                "| Category | Correct | Total | Accuracy |",
                "|----------|---------|-------|----------|",
            ]
            for cat in sorted(per_cat_detail.keys()):
                d = per_cat_detail[cat]
                md_lines.append("| {} | {} | {} | {:.1f}% |".format(
                    cat, d["correct"], d["total"], 100 * d["accuracy"],
                ))
            md_lines.append("")
            (out_dir / "train_by_category.md").write_text("\n".join(md_lines))
            print(f"[Save] Category accuracy → {out_dir / 'train_by_category.md'}")


if __name__ == "__main__":
    main()
