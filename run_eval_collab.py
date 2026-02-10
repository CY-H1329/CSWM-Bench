#!/usr/bin/env python3
"""
Qwen 단일, LLaVA 단일, Qwen+LLaVA 협력(2 agents) 성능 비교.
협력 simple: 의견 일치면 그 답, 불일치면 tie_break.
협력 discuss: 추론 교환 → 상대 추론에 납득하면 그 답, 최대 3라운드 후에도 불일치면 TIE.

Usage:
  python run_eval_collab.py --split train --max_per_category 7
  python run_eval_collab.py --collab_mode discuss   # 추론 교환·동의 후 최종 답
"""
import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm

from src.data import load_stvqa, accuracy, get_prompt, get_prompt_with_reasoning, normalize_answer_only
from run_eval import load_config, run_model
from run_eval_multiagent import get_runner


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


def parse_reasoning_and_answer(text: str):
    """모델 출력에서 추론 문장과 답(A/B/C/D) 추출. (reasoning, answer_letter)"""
    text = (text or "").strip()
    answer = normalize_answer_only(text)
    reasoning = text
    for sep in ["Answer:", "answer:", "Answer ", "answer "]:
        idx = text.find(sep)
        if idx >= 0:
            reasoning = text[:idx].strip()
            break
    # 마지막 (A)/(B)/(C)/(D) 앞까지를 추론으로
    for c in "ABCD":
        for fmt in [f"({c})", f"({c}).", f" {c})"]:
            idx = text.upper().rfind(fmt.upper())
            if idx >= 0:
                reasoning = text[:idx].strip()
                break
    return (reasoning[:500] if len(reasoning) > 500 else reasoning, answer)


def parse_agree_disagree(text: str):
    """AGREE (X) / DISAGREE ... Answer: (Y) 파싱. (agreed, letter) agreed=True면 상대 답 수용."""
    text_raw = (text or "").strip()
    text = text_raw.upper()
    if "AGREE" in text:
        letter = normalize_answer_only(text_raw)
        return True, letter
    letter = normalize_answer_only(text_raw)
    return False, letter


MAX_DISCUSS_ROUNDS = 3


def run_collab_discuss(dataset, config, run_dir: Path, n: int, gt_list: list):
    """
    협의 모드: 각 샘플에 대해 추론 요청 → 불일치 시 상대 추론 제시 후 AGREE/DISAGREE, 최대 3라운드. TIE면 "" 반환.
    반환: (collab_preds, agree_count, tie_count, details_per_sample)
    """
    eval_cfg = config.get("eval", {})
    # 협업 discuss: 추론·답을 다양하게 하기 위해 temperature > 0, top_k/top_p 사용
    temp = eval_cfg.get("collab_temperature") or eval_cfg.get("multi_agent_temperature", 0.4)
    max_new = eval_cfg.get("max_new_tokens", 512)
    top_k = eval_cfg.get("top_k", 50)
    top_p = eval_cfg.get("top_p", 0.9)
    runner_q = get_runner("qwen", config)
    runner_l = get_runner("llava", config)
    if runner_q is None or runner_l is None:
        return None
    print(f"  [collab discuss] temperature={temp}, top_k={top_k}, top_p={top_p} (답 다양화)")

    collab_preds = []
    agree_count = 0
    tie_count = 0
    details = []

    for i in tqdm(range(n), desc="collab_discuss"):
        row = dataset[i]
        img = row.get("images") or row.get("image")
        prompt_r = get_prompt_with_reasoning(row)
        gt = gt_list[i]

        # Round 1: 추론 + 답
        out_q = runner_q.generate(img, prompt_r, temperature=temp, max_new_tokens=max_new, top_k=top_k, top_p=top_p)
        out_l = runner_l.generate(img, prompt_r, temperature=temp, max_new_tokens=max_new, top_k=top_k, top_p=top_p)
        r_q, a_q = parse_reasoning_and_answer(out_q)
        r_l, a_l = parse_reasoning_and_answer(out_l)

        detail = {"idx": i, "round1": {"qwen": {"reasoning": r_q, "answer": a_q}, "llava": {"reasoning": r_l, "answer": a_l}}}

        if a_q == a_l and a_q:
            collab_preds.append(a_q)
            agree_count += 1
            detail["final"] = a_q
            detail["reason"] = "agree"
            details.append(detail)
            continue

        # Round 2, 3: 상대 추론 제시 후 AGREE/DISAGREE
        final = ""
        for round_no in range(2, MAX_DISCUSS_ROUNDS + 1):
            other_l = a_l if a_l in "ABCD" else "X"
            other_q = a_q if a_q in "ABCD" else "X"
            prompt_q = (
                f"The other agent's reasoning: {r_l}\nThe other agent's answer: ({other_l}).\n"
                "If you are convinced, reply exactly: AGREE (" + other_l + "). "
                "If not, reply: DISAGREE. Then write Reasoning: [why] and Answer: (A/B/C/D)."
            )
            prompt_l = (
                f"The other agent's reasoning: {r_q}\nThe other agent's answer: ({other_q}).\n"
                "If you are convinced, reply exactly: AGREE (" + other_q + "). "
                "If not, reply: DISAGREE. Then write Reasoning: [why] and Answer: (A/B/C/D)."
            )
            out_q2 = runner_q.generate(img, prompt_q, temperature=temp, max_new_tokens=max_new, top_k=top_k, top_p=top_p)
            out_l2 = runner_l.generate(img, prompt_l, temperature=temp, max_new_tokens=max_new, top_k=top_k, top_p=top_p)
            agree_q, letter_q = parse_agree_disagree(out_q2)
            agree_l, letter_l = parse_agree_disagree(out_l2)
            detail[f"round{round_no}"] = {"qwen_agree": agree_q, "qwen_letter": letter_q, "llava_agree": agree_l, "llava_letter": letter_l}

            if agree_q and letter_q in "ABCD":
                final = letter_q
                detail["final"] = final
                detail["reason"] = f"qwen_agreed_round{round_no}"
                break
            if agree_l and letter_l in "ABCD":
                final = letter_l
                detail["final"] = final
                detail["reason"] = f"llava_agreed_round{round_no}"
                break
            if letter_q == letter_l and letter_q:
                final = letter_q
                detail["final"] = final
                detail["reason"] = f"both_same_round{round_no}"
                break
            r_q, a_q = parse_reasoning_and_answer(out_q2)
            r_l, a_l = parse_reasoning_and_answer(out_l2)

        if not final:
            tie_count += 1
            detail["final"] = "TIE"
            detail["reason"] = "tie_after_3_rounds"
            collab_preds.append("")  # TIE → 정답 비교 시 wrong 처리
        else:
            collab_preds.append(final)
        details.append(detail)

    return collab_preds, agree_count, tie_count, details


def main():
    parser = argparse.ArgumentParser(description="Qwen vs LLaVA vs Qwen+LLaVA 협력 비교")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_category", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tie_break", default="qwen", choices=["qwen", "llava"],
                        help="simple 모드에서 불일치 시 쓸 쪽")
    parser.add_argument("--collab_mode", default="simple", choices=["simple", "discuss"],
                        help="simple=일치면 그 답/tie_break. discuss=추론 교환 후 동의하면 그 답, 3라운드 후에도 불일치면 TIE")
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
    print(f"Collaboration: Qwen + LLaVA (2 agents). Mode: {args.collab_mode}" + (f", tie_break={args.tie_break}" if args.collab_mode == "simple" else " (추론 교환, 최대 3라운드, TIE 처리)"))

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

    # 2) 협력 (Qwen + LLaVA)
    qwen_recs = load_preds_jsonl(run_dir / "qwen_preds.jsonl")
    llava_recs = load_preds_jsonl(run_dir / "llava_preds.jsonl")
    n = len(dataset)
    gt_list = [dataset[i]["answer_only"] for i in range(n)]
    collab_preds = []
    agree_count = 0
    tie_count = 0
    discuss_details = None

    if args.collab_mode == "simple":
        for i in range(n):
            q = qwen_recs.get(i, {}).get("pred", "")
            l = llava_recs.get(i, {}).get("pred", "")
            if q == l:
                collab_preds.append(q)
                agree_count += 1
            else:
                collab_preds.append(q if args.tie_break == "qwen" else l)
        print(f"\n--- 2) Collaboration (Qwen + LLaVA), tie_break={args.tie_break} ---")
        print(f"  Agree: {agree_count}/{n},  Collab accuracy = {accuracy(collab_preds, gt_list):.4f}")
    else:
        result = run_collab_discuss(dataset, config, run_dir, n, gt_list)
        if result is None:
            raise RuntimeError("discuss mode requires qwen and llava runners")
        collab_preds, agree_count, tie_count, discuss_details = result
        with open(run_dir / "collab_discuss_details.jsonl", "w") as f:
            for d in discuss_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_correct = sum(1 for i in range(n) if collab_preds[i] == gt_list[i])
        print(f"\n--- 2) Collaboration (Qwen + LLaVA), discuss mode ---")
        print(f"  Agree (1r): {agree_count}/{n},  TIE (3r 후): {tie_count},  Collab accuracy = {n_correct}/{n} = {n_correct/n:.4f}")

    collab_acc = accuracy(collab_preds, gt_list)

    # 협력 결과 저장
    collab_results = {
        "model": "qwen_llava_collab",
        "accuracy": collab_acc,
        "num_samples": n,
        "collab_mode": args.collab_mode,
        "tie_break": args.tie_break if args.collab_mode == "simple" else None,
        "agree_count": agree_count,
        "tie_count": tie_count if args.collab_mode == "discuss" else None,
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
            qp = qwen_recs.get(i, {}).get("pred")
            lp = llava_recs.get(i, {}).get("pred")
            rec = {
                "idx": i,
                "qwen_pred": qp,
                "llava_pred": lp,
                "collab_pred": collab_preds[i] if i < len(collab_preds) else "",
                "gt": gt_list[i],
                "correct": (collab_preds[i] == gt_list[i]) if i < len(collab_preds) else False,
                "agree": (qp == lp) if args.collab_mode == "simple" else None,
                "tie": (args.collab_mode == "discuss" and (collab_preds[i] == "" if i < len(collab_preds) else False)),
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
        {"setting": "qwen_llava_collab", "accuracy": collab_acc, "num_samples": n, "collab_mode": args.collab_mode, "tie_break": args.tie_break if args.collab_mode == "simple" else None, "tie_count": tie_count if args.collab_mode == "discuss" else None},
    ]
    with open(run_dir / "comparison_collab.json", "w") as f:
        json.dump(comparison, f, indent=2)

    lines = [
        "=== Qwen vs LLaVA vs Qwen+LLaVA 협력 ===",
        f"Mode: {args.collab_mode}" + (f", tie_break={args.tie_break}" if args.collab_mode == "simple" else f", TIE={tie_count}"),
        "",
        "setting              | accuracy  | n",
        "-" * 40,
        f"qwen_only            | {(f'{qwen_acc:.2%}' if qwen_acc is not None else 'N/A'):10} | {n}",
        f"llava_only           | {(f'{llava_acc:.2%}' if llava_acc is not None else 'N/A'):10} | {n}",
        f"qwen_llava_collab    | {collab_acc:.2%}  ({args.collab_mode}) | {n}",
        "",
        f"Agree (1라운드 일치): {agree_count}/{n}" + (f",  TIE (3r 후): {tie_count}" if args.collab_mode == "discuss" else ""),
    ]
    (run_dir / "comparison_collab.txt").write_text("\n".join(lines), encoding="utf-8")

    with open(run_dir / "comparison_collab.csv", "w") as f:
        f.write("setting,accuracy,num_samples\n")
        for c in comparison:
            acc = c.get("accuracy")
            acc_str = f"{acc:.4f}" if acc is not None else ""
            f.write(f"{c['setting']},{acc_str},{n}\n")

    print("\n--- 3) Comparison ---")
    print(f"  qwen_only         : {qwen_acc:.2%}" if qwen_acc is not None else "  qwen_only         : N/A")
    print(f"  llava_only        : {llava_acc:.2%}" if llava_acc is not None else "  llava_only        : N/A")
    print(f"  qwen_llava_collab : {collab_acc:.2%} (tie_break={args.tie_break})")
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
