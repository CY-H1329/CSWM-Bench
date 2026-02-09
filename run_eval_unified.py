#!/usr/bin/env python3
"""
단일 에이전트 → 멀티 에이전트(3명 다수결)를 한 프로그램에서 순차 실행.
정확도 비교 + 서로 틀린 문제 비교(둘 다 틀림 / 단일만 틀림·멀티만 맞음 / 멀티만 틀림 등)까지 한 번에 정리.

Usage:
  python run_eval_unified.py --models qwen llava --split train --max_per_category 100
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml

from src.data import load_stvqa
from run_eval import load_config, run_model
from run_eval_multiagent import run_model_multiagent


def load_preds_jsonl(path: Path) -> list:
    """jsonl 한 줄씩 로드. idx, correct (및 기타) 필드 사용."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compare_wrong_problems(run_dir: Path, model_name: str) -> dict | None:
    """
    단일 preds vs 멀티 preds 로 틀린 문제 비교.
    반환: both_wrong, only_single_wrong (multi가 맞춤=회복), only_multi_wrong (단일은 맞춤=악화), counts
    """
    single_path = run_dir / f"{model_name}_preds.jsonl"
    multi_path = run_dir / f"{model_name}_multiagent_preds.jsonl"
    if not single_path.exists() or not multi_path.exists():
        return None
    single_recs = {r["idx"]: r for r in load_preds_jsonl(single_path)}
    multi_recs = {r["idx"]: r for r in load_preds_jsonl(multi_path)}
    single_wrong = {idx for idx, r in single_recs.items() if not r.get("correct", True)}
    multi_wrong = {idx for idx, r in multi_recs.items() if not r.get("correct", True)}
    both_wrong = single_wrong & multi_wrong
    only_single_wrong = single_wrong - multi_wrong  # multi가 맞춤 (회복)
    only_multi_wrong = multi_wrong - single_wrong   # 단일은 맞았는데 multi가 틀림 (악화)
    both_correct = set(single_recs) - single_wrong - only_multi_wrong  # 둘 다 맞음
    n = len(single_recs)
    return {
        "model": model_name,
        "num_samples": n,
        "single_wrong_count": len(single_wrong),
        "multi_wrong_count": len(multi_wrong),
        "both_wrong_count": len(both_wrong),
        "only_single_wrong_count": len(only_single_wrong),
        "only_multi_wrong_count": len(only_multi_wrong),
        "both_correct_count": len(both_correct),
        "recovered_by_multi": len(only_single_wrong),
        "regressed_in_multi": len(only_multi_wrong),
        "indices": {
            "both_wrong": sorted(both_wrong),
            "only_single_wrong": sorted(only_single_wrong),
            "only_multi_wrong": sorted(only_multi_wrong),
        },
    }


def write_wrong_comparison_report(run_dir: Path, models: list, comparisons: list):
    """틀린 문제 비교 결과를 JSON + 텍스트로 저장."""
    with open(run_dir / "wrong_comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    lines = [
        "=== 단일 vs 멀티(3 agents) 틀린 문제 비교 ===",
        "",
        "용어:",
        "  both_wrong       : 단일·멀티 둘 다 틀린 문제",
        "  only_single_wrong: 단일만 틀림 → 멀티는 맞음 (다수결로 회복)",
        "  only_multi_wrong : 단일은 맞음 → 멀티만 틀림 (악화)",
        "",
    ]
    for c in comparisons:
        m = c["model"]
        n = c["num_samples"]
        sw = c["single_wrong_count"]
        mw = c["multi_wrong_count"]
        both = c["both_wrong_count"]
        only_s = c["only_single_wrong_count"]
        only_m = c["only_multi_wrong_count"]
        lines.extend([
            f"--- {m} (n={n}) ---",
            f"  단일 틀린 개수: {sw}",
            f"  멀티 틀린 개수: {mw}",
            f"  둘 다 틀림:     {both}",
            f"  단일만 틀림 (멀티 회복): {only_s}",
            f"  멀티만 틀림 (악화):     {only_m}",
            "",
        ])
    (run_dir / "wrong_comparison.txt").write_text("\n".join(lines), encoding="utf-8")

    # CSV: 모델별 요약
    with open(run_dir / "wrong_comparison.csv", "w") as f:
        f.write("model,num_samples,single_wrong,multi_wrong,both_wrong,recovered_by_multi,regressed_in_multi\n")
        for c in comparisons:
            f.write(f"{c['model']},{c['num_samples']},{c['single_wrong_count']},{c['multi_wrong_count']},"
                    f"{c['both_wrong_count']},{c['recovered_by_multi']},{c['regressed_in_multi']}\n")


def main():
    parser = argparse.ArgumentParser(description="단일 → 멀티 순차 실행 + 정확도·틀린문제 비교")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=["qwen", "llava"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_category", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    if args.max_samples is not None:
        ds_cfg = {**ds_cfg, "max_samples": args.max_samples}
    max_per_cat = args.max_per_category or ds_cfg.get("max_per_category")

    # 예측 저장·카테고리 정확도 강제
    if "output" not in config:
        config["output"] = {}
    config["output"]["save_predictions"] = True
    config["output"]["per_category_accuracy"] = True

    dataset = load_stvqa(
        dataset_name=ds_cfg.get("name", "OX-PIXL/STVQA-7K"),
        split=args.split,
        max_samples=ds_cfg.get("max_samples"),
        max_per_category=max_per_cat,
    )
    suffix = f", max_per_category={max_per_cat})" if max_per_cat else ""
    print(f"Loaded {len(dataset)} samples (split={args.split}{suffix})")

    output_dir = Path(args.output_dir or config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_unified"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f)
    with open(run_dir / "dataset_info.json", "w") as f:
        json.dump({
            "dataset_name": ds_cfg.get("name", "OX-PIXL/STVQA-7K"),
            "split": args.split,
            "max_samples": ds_cfg.get("max_samples"),
            "max_per_category": max_per_cat,
            "num_loaded": len(dataset),
        }, f, indent=2)

    # 1) 단일 에이전트
    print("\n--- 1) Single-agent ---")
    single_results = []
    for model_name in args.models:
        try:
            res = run_model(model_name, dataset, config, run_dir)
            if res is not None:
                single_results.append(res)
                print(f"  {model_name}: accuracy = {res['accuracy']:.4f} ({res['num_samples']} samples)")
        except Exception as e:
            print(f"  {model_name}: error - {e}")
            raise

    # 2) 멀티 에이전트 (3명 다수결)
    print("\n--- 2) Multi-agent (3 agents, majority vote) ---")
    multi_results = []
    for model_name in args.models:
        try:
            res = run_model_multiagent(model_name, dataset, config, run_dir)
            if res is not None:
                multi_results.append(res)
                print(f"  {model_name}: accuracy = {res['accuracy']:.4f} ({res['num_samples']} samples)")
        except Exception as e:
            print(f"  {model_name}: error - {e}")
            raise

    # 3) 정확도 비교
    print("\n--- 3) Accuracy comparison (single vs multi) ---")
    comparison = []
    for sr in single_results:
        m = sr["model"]
        multi_acc = next((r["accuracy"] for r in multi_results if r["model"] == m), None)
        single_acc = sr["accuracy"]
        delta = (multi_acc - single_acc) if multi_acc is not None else None
        comparison.append({
            "model": m,
            "single_agent_accuracy": single_acc,
            "multi_agent_accuracy": multi_acc,
            "delta": round(delta, 4) if delta is not None else None,
        })
        d_str = f"{delta:+.2%}" if delta is not None else "N/A"
        print(f"  {m}: single {single_acc:.2%}  multi {multi_acc:.2%}  delta {d_str}")

    with open(run_dir / "comparison_single_vs_multi.json", "w") as f:
        json.dump(comparison, f, indent=2)
    comp_lines = [
        "model          | single (1 agent) | multi (3 agents) | delta",
        "-" * 60,
    ]
    for c in comparison:
        s = f"{c['single_agent_accuracy']:.2%}"
        mu = f"{c['multi_agent_accuracy']:.2%}" if c["multi_agent_accuracy"] is not None else "N/A"
        d = f"{c['delta']:+.2%}" if c["delta"] is not None else "N/A"
        comp_lines.append(f"{c['model']:14} | {s:16} | {mu:16} | {d}")
    (run_dir / "comparison_single_vs_multi.txt").write_text("\n".join(comp_lines), encoding="utf-8")
    with open(run_dir / "comparison_single_vs_multi.csv", "w") as f:
        f.write("model,single_agent_accuracy,multi_agent_accuracy,delta\n")
        for c in comparison:
            f.write(f"{c['model']},{c['single_agent_accuracy']:.4f},{c.get('multi_agent_accuracy') or ''},{c.get('delta') or ''}\n")

    # 4) 틀린 문제 비교
    print("\n--- 4) Wrong-problem comparison ---")
    wrong_comparisons = []
    for model_name in args.models:
        cmp = compare_wrong_problems(run_dir, model_name)
        if cmp:
            wrong_comparisons.append(cmp)
            print(f"  {model_name}: single_wrong={cmp['single_wrong_count']}  multi_wrong={cmp['multi_wrong_count']}  "
                  f"both_wrong={cmp['both_wrong_count']}  recovered={cmp['recovered_by_multi']}  regressed={cmp['regressed_in_multi']}")
    if wrong_comparisons:
        write_wrong_comparison_report(run_dir, args.models, wrong_comparisons)
        print(f"  Saved: wrong_comparison.json, wrong_comparison.txt, wrong_comparison.csv")

    print(f"\nResults saved to {run_dir}")
    print(f"  Conversations: {run_dir / 'conversations'}")


if __name__ == "__main__":
    main()
