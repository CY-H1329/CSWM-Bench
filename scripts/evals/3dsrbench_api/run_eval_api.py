#!/usr/bin/env python3
"""
3DSRBench & CV-Bench — API models (Claude Sonnet 4.5, GPT-4o, GPT-5.2, DeepSeek-VL, Gemini Robotics-ER).

Usage:
  python scripts/evals/3dsrbench_api/run_eval_api.py
  python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench --max_samples 200
  python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench --full_dataset
  # Full HF test split (not frozen): --full_dataset sets use_frozen=False
  python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt_5_2
  python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset   # all enabled models
  python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
  python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt
  python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 1000 --model claude_sonnet_4_5 --without_prompt --start_idx 466 --end_idx 1000 --resume_dir results/runs/3dsrbench/api_models/20260217_060437/claude_sonnet_4_5_without_prompt  # reprise 466-999

Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from src.benchmarks import (
    load_benchmark,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
)
from src.data import (
    normalize_answer_only,
    accuracy,
    extract_predicted_category,
    normalize_category,
    CV_BENCH_CLASSIFICATION_CATS,
)

# Import prompt from common (sibling)
import importlib.util
_common_path = ROOT / "scripts/evals/3dsrbench/common.py"
_spec = importlib.util.spec_from_file_location("common", _common_path)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
build_spatial_prompt = _common.build_spatial_prompt
build_cvbench_prompt = _common.build_cvbench_prompt

# Import API runners (same directory)
_runners_path = Path(__file__).parent / "runners.py"
_runners_spec = importlib.util.spec_from_file_location("runners", _runners_path)
_runners = importlib.util.module_from_spec(_runners_spec)
_runners_spec.loader.exec_module(_runners)
ClaudeRunner = _runners.ClaudeRunner
GPT4oRunner = _runners.GPT4oRunner
DeepSeekVLRunner = _runners.DeepSeekVLRunner
GeminiRunner = _runners.GeminiRunner


def get_runner(model_key: str, config: dict):
    cfg = config.get("models", {}).get(model_key, {})
    if not cfg.get("enabled", True):
        return None
    api_key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key:
        return None
    model_id = cfg.get("model_id", "")
    if model_key == "claude_sonnet_4_5":
        return ClaudeRunner(model_id=model_id, api_key=api_key)
    if model_key in ("gpt4o", "gpt_5_2"):
        return GPT4oRunner(model_id=model_id, api_key=api_key)
    if model_key == "deepseek_vl":
        return DeepSeekVLRunner(
            model_id=model_id,
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
        )
    if model_key == "gemini_robotics_er":
        return GeminiRunner(model_id=model_id, api_key=api_key)
    return None


def _write_details_checkpoint(model_dir: Path, details: list) -> None:
    """상용: 중간 저장 (중단 시 resume_dir + start_idx로 이어가기)."""
    path = model_dir / "details.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _per_category_answer_accuracy(details: list, preds: list, gt_list: list) -> dict:
    """Answer accuracy (MCQ letter) grouped by benchmark category / task."""
    from collections import defaultdict

    bucket = defaultdict(lambda: [0, 0])  # correct, total
    for d, p, g in zip(details, preds, gt_list):
        cat = d.get("category")
        if not cat:
            cat = d.get("category_gt") or "unknown"
        bucket[cat][1] += 1
        if p == g:
            bucket[cat][0] += 1
    out = {}
    for cat in sorted(bucket.keys(), key=lambda x: (x == "unknown", str(x).lower())):
        c, t = bucket[cat]
        out[str(cat)] = {"n": t, "correct": c, "accuracy": (c / t) if t else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser(description="3DSRBench / CV-Bench API models")
    parser.add_argument(
        "--benchmark",
        choices=["3dsrbench", "cvbench"],
        default="3dsrbench",
        help="Frozen local split by default; --full_dataset loads full HF test",
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_api.yaml"))
    parser.add_argument("--max_samples", type=int, default=None, help="Limiter à N samples (défaut: 1000 si pas --full_dataset)")
    parser.add_argument("--full_dataset", action="store_true", help="Dataset complet, sortie dans full_dataset/")
    parser.add_argument(
        "--model",
        choices=["claude_sonnet_4_5", "gpt4o", "gpt_5_2", "gemini_robotics_er"],
        help="Un seul modèle (pour terminaux séparés)",
    )
    parser.add_argument("--without_prompt", action="store_true", help="Question seule (un seul run)")
    parser.add_argument("--prompt_variant", choices=["with_prompt", "without_prompt"], help="Une seule variante (exclut l'autre)")
    parser.add_argument("--start_idx", type=int, default=None, help="Reprise : indices à partir de (ex: 466)")
    parser.add_argument("--end_idx", type=int, default=None, help="Reprise : indices jusqu'à exclu (ex: 1000)")
    parser.add_argument("--resume_dir", default=None, help="Reprise : dossier du run existant (ex: .../claude_sonnet_4_5_without_prompt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Write details.jsonl every N samples (0 = only at end).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    bench_name = args.benchmark
    ds_cfg = config.get("dataset", {})
    use_full = args.full_dataset
    max_samples = None if use_full else (args.max_samples or ds_cfg.get("max_samples", 1000))
    seed = args.seed or ds_cfg.get("seed", 42)
    output_dir = Path(config.get("output", {}).get("dir", "results"))
    prompt_with_ctx = build_cvbench_prompt if bench_name == "cvbench" else build_spatial_prompt
    valid_cls_cats = CV_BENCH_CLASSIFICATION_CATS if bench_name == "cvbench" else None

    use_resume = args.resume_dir and args.start_idx is not None and args.end_idx is not None
    resume_path = Path(args.resume_dir).resolve() if args.resume_dir else None
    if use_resume:
        run_dir = resume_path.parent
    else:
        subdir = "full_dataset" if use_full else datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / "runs" / bench_name / "api_models" / subdir
        run_dir.mkdir(parents=True, exist_ok=True)

    if use_resume:
        start_idx = args.start_idx
        end_idx = args.end_idx
    else:
        start_idx = 0
        end_idx = None  # set after load

    bench_label = "CV-Bench" if bench_name == "cvbench" else "3DSRBench"
    print(
        f"Loading {bench_label}... (max_samples={'all' if use_full else max_samples}, "
        f"seed={seed}, use_frozen={not use_full})"
    )
    dataset = load_benchmark(
        bench_name,
        max_samples=max_samples,
        seed=seed,
        use_frozen=not use_full,
    )
    print(f"  {len(dataset)} samples")
    if use_resume:
        print(f"  Reprise: indices {start_idx} à {end_idx - 1} ({end_idx - start_idx} samples)")
    else:
        end_idx = len(dataset)
    if use_resume:
        if "claude" in resume_path.name:
            model_keys = ["claude_sonnet_4_5"]
        elif "gpt4o" in resume_path.name:
            model_keys = ["gpt4o"]
        elif "gpt_5_2" in resume_path.name:
            model_keys = ["gpt_5_2"]
        elif "gemini" in resume_path.name:
            model_keys = ["gemini_robotics_er"]
        elif "deepseek" in resume_path.name:
            model_keys = ["deepseek_vl"]
        else:
            model_keys = ["claude_sonnet_4_5"]
        variant_name = "without_prompt" if "without" in resume_path.name else "with_prompt"
        prompt_fn = (
            (lambda q: q)
            if variant_name == "without_prompt"
            else (lambda q: prompt_with_ctx(q))
        )
        prompt_variants = [(variant_name, prompt_fn)]
    else:
        model_keys = (
            [args.model]
            if args.model
            else ["claude_sonnet_4_5", "gpt4o", "gpt_5_2", "deepseek_vl", "gemini_robotics_er"]
        )
        if args.prompt_variant == "with_prompt":
            prompt_variants = [("with_prompt", lambda q: prompt_with_ctx(q))]
        elif args.without_prompt or args.prompt_variant == "without_prompt":
            prompt_variants = [("without_prompt", lambda q: q)]
        else:
            prompt_variants = [
                ("with_prompt", lambda q: prompt_with_ctx(q)),
                ("without_prompt", lambda q: q),
            ]
    results_table = []

    for model_key in model_keys:
        runner = get_runner(model_key, config)
        if runner is None:
            print(f"\n[skip] {model_key} (disabled or missing API key)")
            continue

        for variant_name, prompt_fn in prompt_variants:
            run_key = f"{model_key}_{variant_name}" if not use_resume else resume_path.name
            print(f"\n--- {run_key} ---")
            if not use_resume:
                model_dir = run_dir / run_key
            else:
                model_dir = resume_path
            model_dir.mkdir(parents=True, exist_ok=True)
            responses_dir = model_dir / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)

            preds = []
            gt_list = []
            details = []
            if use_resume:
                existing_details_path = model_dir / "details.jsonl"
                if existing_details_path.exists():
                    with open(existing_details_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                d = json.loads(line)
                                if d.get("idx", -1) < start_idx:
                                    details.append(d)
                                    preds.append(d.get("pred", ""))
                                    gt_list.append(d.get("gt", ""))
                    print(f"  Chargé {len(details)} samples existants (idx < {start_idx})")
                else:
                    print(f"  [ERREUR] details.jsonl absent. Exécutez d'abord recover:")
                    print(f"    python scripts/evals/3dsrbench_api/recover_from_responses.py --dir {model_dir}")
                    sys.exit(1)

            indices = list(range(start_idx, min(end_idx, len(dataset))))
            for i in tqdm(indices, desc=run_key):
                example = dataset[i]
                image = get_benchmark_image(example, bench_name)
                query = get_benchmark_prompt(example, bench_name)
                gt = get_benchmark_answer(example, bench_name)
                category = get_benchmark_category(example, bench_name) or "unknown"
                gt_category_norm = normalize_category(category) if category and category != "unknown" else ""

                if image is None:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({
                        "idx": i,
                        "error": "no_image",
                        "gt": gt,
                        "category": category,
                        "category_gt": gt_category_norm,
                    })
                    continue

                full_prompt = prompt_fn(query)

                try:
                    response = runner.generate(
                        image,
                        full_prompt,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    )
                    letter = normalize_answer_only(response)
                    pred_category = extract_predicted_category(response, valid_cats=valid_cls_cats)
                    preds.append(letter)
                    gt_list.append(gt)
                    details.append({
                        "idx": i,
                        "query": query,
                        "gt": gt,
                        "pred": letter,
                        "category": category,
                        "category_gt": gt_category_norm,
                        "pred_category": pred_category,
                        "full_response": response,
                    })
                    with open(responses_dir / f"sample_{i:05d}.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== QUERY ===\n{query}\n\n=== GT ===\n{gt}\n\n")
                        f.write(f"=== CATEGORY GT / PRED ===\n{gt_category_norm} / {pred_category}\n\n")
                        f.write(f"=== FULL RESPONSE ===\n{response}\n\n=== EXTRACTED PRED ===\n{letter}\n")
                except Exception as e:
                    preds.append("")
                    gt_list.append(gt)
                    details.append({
                        "idx": i,
                        "error": str(e),
                        "gt": gt,
                        "category": category,
                        "category_gt": gt_category_norm,
                    })

                if args.checkpoint_every and len(details) % args.checkpoint_every == 0:
                    _write_details_checkpoint(model_dir, details)

            if use_resume and details:
                details.sort(key=lambda d: d.get("idx", 0))
            acc = accuracy(preds, gt_list)
            cat_pairs = [(d.get("category_gt", ""), d.get("pred_category", "")) for d in details if d.get("category_gt")]
            cat_cls_acc = accuracy([p[1] for p in cat_pairs], [p[0] for p in cat_pairs]) if cat_pairs else 0.0
            pred_dist = {k: v for k, v in sorted(Counter(p for p in preds if p).items())}
            per_cat = _per_category_answer_accuracy(details, preds, gt_list)

            with open(model_dir / "details.jsonl", "w", encoding="utf-8") as f:
                for d in details:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            with open(model_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark": bench_name,
                    "model": run_key,
                    "prompt_variant": variant_name,
                    "accuracy": acc,
                    "n": len(details),
                    "pred_distribution": pred_dist,
                    "category_cls_accuracy": cat_cls_acc,
                    "category_cls_n": len(cat_pairs),
                    "per_category_answer_accuracy": per_cat,
                }, f, indent=2)

            results_table.append({
                "model": run_key,
                "accuracy": acc,
                "category_cls_acc": cat_cls_acc,
                "n": len(details),
                "per_category": per_cat,
            })
            print(f"  Answer Accuracy: {acc:.4f} | Category Cls: {cat_cls_acc:.4f} | N={len(details)}")
            print("  Per-category (answer acc):")
            for cname, st in per_cat.items():
                print(f"    {cname}: {st['accuracy']:.4f} ({st['correct']}/{st['n']})")

    # Summary (sauf si run unique --model : évite écrasement en parallèle)
    n_total = len(dataset)
    if not args.model or len(results_table) > 1:
        with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"# {bench_label} — API Models ({n_total} samples)\n")
            f.write(f"# benchmark={bench_name}\n")
            f.write("# Chaque modèle : with_prompt (spatial) et without_prompt (question seule)\n\n")
            f.write("| Model | Answer Acc | Category Cls Acc | N |\n")
            f.write("|-------|------------|------------------|---|\n")
            for r in results_table:
                f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['category_cls_acc']:.4f} | {r['n']} |\n")
            f.write("\n## Per-category answer accuracy\n\n")
            for r in results_table:
                f.write(f"### {r['model']}\n\n")
                f.write("| Category | Accuracy | Correct | N |\n")
                f.write("|----------|----------|---------|---|\n")
                for cname, st in r["per_category"].items():
                    f.write(
                        f"| {cname} | {st['accuracy']:.4f} | {st['correct']} | {st['n']} |\n"
                    )
                f.write("\n")

    print("\n" + "=" * 60)
    print(f"{bench_label} — API Models Summary")
    print("=" * 60)
    for r in results_table:
        print(f"  {r['model']}: Answer={r['accuracy']:.4f}, CatCls={r['category_cls_acc']:.4f}")
        for cname, st in r["per_category"].items():
            print(f"      [{cname}] {st['accuracy']:.4f} ({st['correct']}/{st['n']})")
    print("=" * 60)
    print(f"Résultats: {run_dir}")


if __name__ == "__main__":
    main()
