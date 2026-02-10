#!/usr/bin/env python3
"""
Qwen 단일, LLaVA 단일, Qwen+LLaVA 협력(2 agents) 성능 비교.
협력: 두 모델이 같은 문제에 각자 답 → 의견 일치면 그 답, 불일치면 tie_break 모델 답 사용.

Usage:
  python run_eval_collab.py --split train --max_per_category 7
  python run_eval_collab.py --tie_break llava   # 불일치 시 LLaVA 답 사용 (기본: qwen)
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml

from src.data import load_stvqa, accuracy
from run_eval import load_config, run_model


def load_preds_jsonl(path: Path) -> dict:
    """idx -> {pred, gt, correct, ...}"""
    if not path.exists():
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def main():
    parser = argparse.ArgumentParser(description="Qwen vs LLaVA vs Qwen+LLaVA 협력 비교")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_category", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tie_break", default="qwen", choices=["qwen", "llava"],
                        help="두 모델 답이 다를 때 쓸 쪽 (기본: qwen)")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    if args.max_samples is not None:
        ds_cfg = {**ds_cfg, "max_samples": args.max_samples}
    max_per_cat = args.max_per_category or ds_cfg.get("max_per_category")

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
    print(f"Collaboration: Qwen + LLaVA (2 agents). Tie-break when disagree: {args.tie_break}")

    output_dir = Path(args.output_dir or config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_collab"
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

    models = ["qwen", "llava"]
    # 1) Qwen 단일, LLaVA 단일
    print("\n--- 1) Single agents ---")
    for model_name in models:
        res = run_model(model_name, dataset, config, run_dir)
        if res is not None:
            print(f"  {model_name}: accuracy = {res['accuracy']:.4f} ({res['num_samples']} samples)")

    # 2) 협력 (Qwen + LLaVA): 일치하면 그 답, 불일치면 tie_break
    qwen_recs = load_preds_jsonl(run_dir / "qwen_preds.jsonl")
    llava_recs = load_preds_jsonl(run_dir / "llava_preds.jsonl")
    n = len(dataset)
    gt_list = [dataset[i]["answer_only"] for i in range(n)]
    collab_preds = []
    agree_count = 0
    for i in range(n):
        q = qwen_recs.get(i, {}).get("pred", "")
        l = llava_recs.get(i, {}).get("pred", "")
        if q == l:
            collab_preds.append(q)
            agree_count += 1
        else:
            collab_preds.append(q if args.tie_break == "qwen" else l)
    collab_acc = accuracy(collab_preds, gt_list)
    print(f"\n--- 2) Collaboration (Qwen + LLaVA), tie_break={args.tie_break} ---")
    print(f"  Agree: {agree_count}/{n},  Collab accuracy = {collab_acc:.4f}")

    # 협력 결과 저장
    collab_results = {
        "model": "qwen_llava_collab",
        "accuracy": collab_acc,
        "num_samples": n,
        "tie_break": args.tie_break,
        "agree_count": agree_count,
    }
    if "category" in dataset.features:
        by_cat = {}
        for i in range(n):
            c = dataset[i]["category"]
            if c not in by_cat:
                by_cat[c] = {"pred": [], "gt": []}
            by_cat[c]["pred"].append(collab_preds[i])
            by_cat[c]["gt"].append(gt_list[i])
        collab_results["by_category"] = {c: accuracy(d["pred"], d["gt"]) for c, d in by_cat.items()}
    with open(run_dir / "collab_qwen_llava_results.json", "w") as f:
        json.dump(collab_results, f, indent=2)

    with open(run_dir / "collab_qwen_llava_preds.jsonl", "w") as f:
        for i in range(n):
            rec = {
                "idx": i,
                "qwen_pred": qwen_recs.get(i, {}).get("pred"),
                "llava_pred": llava_recs.get(i, {}).get("pred"),
                "collab_pred": collab_preds[i],
                "gt": gt_list[i],
                "correct": collab_preds[i] == gt_list[i],
                "agree": qwen_recs.get(i, {}).get("pred") == llava_recs.get(i, {}).get("pred"),
                "category": dataset[i].get("category"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) 세 가지 비교
    qwen_acc = None
    llava_acc = None
    for p in [run_dir / "qwen_results.json", run_dir / "llava_results.json"]:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            if d.get("model") == "qwen":
                qwen_acc = d["accuracy"]
            elif d.get("model") == "llava":
                llava_acc = d["accuracy"]

    comparison = [
        {"setting": "qwen_only", "accuracy": qwen_acc, "num_samples": n},
        {"setting": "llava_only", "accuracy": llava_acc, "num_samples": n},
        {"setting": "qwen_llava_collab", "accuracy": collab_acc, "num_samples": n, "tie_break": args.tie_break},
    ]
    with open(run_dir / "comparison_collab.json", "w") as f:
        json.dump(comparison, f, indent=2)

    lines = [
        "=== Qwen vs LLaVA vs Qwen+LLaVA 협력 ===",
        f"Tie-break (불일치 시): {args.tie_break}",
        "",
        "setting              | accuracy  | n",
        "-" * 40,
        f"qwen_only            | {(f'{qwen_acc:.2%}' if qwen_acc is not None else 'N/A'):10} | {n}",
        f"llava_only           | {(f'{llava_acc:.2%}' if llava_acc is not None else 'N/A'):10} | {n}",
        f"qwen_llava_collab    | {collab_acc:.2%}  (tie_break={args.tie_break}) | {n}",
        "",
        f"Agree (두 모델 답 일치): {agree_count}/{n}",
    ]
    (run_dir / "comparison_collab.txt").write_text("\n".join(lines), encoding="utf-8")

    with open(run_dir / "comparison_collab.csv", "w") as f:
        f.write("setting,accuracy,num_samples\n")
        for c in comparison:
            acc = c.get("accuracy")
            f.write(f"{c['setting']},{acc:.4f if acc is not None else ''},{n}\n")

    print("\n--- 3) Comparison ---")
    print(f"  qwen_only         : {qwen_acc:.2%}" if qwen_acc is not None else "  qwen_only         : N/A")
    print(f"  llava_only        : {llava_acc:.2%}" if llava_acc is not None else "  llava_only        : N/A")
    print(f"  qwen_llava_collab : {collab_acc:.2%} (tie_break={args.tie_break})")
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
