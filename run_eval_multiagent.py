#!/usr/bin/env python3
"""
Multi-agent 평가: 동일 모델을 3명의 에이전트로 호출 → 같은 문제에 각자 답 → 다수결로 최종 답.
단일 에이전트 run과 비교해 수치화 (single acc vs multi acc, delta).

Usage:
  python run_eval_multiagent.py --models qwen llava --split train --max_per_category 100
  python run_eval_multiagent.py --models qwen llava --baseline_run_dir results/20260209_175745
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

import yaml
from tqdm import tqdm

from src.data import (
    load_stvqa,
    get_prompt,
    normalize_answer_only,
    accuracy,
)
from src.models.qwen import QwenRunner
from src.models.llava import LLaVARunner
from src.models.gpt import GPTRunner
from src.models.gemini import GeminiRunner


NUM_AGENTS = 3


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def majority_vote(letters: list) -> str:
    """다수결. 동점이면 유효한 답 중 첫 번째 에이전트 답 사용."""
    valid = [x for x in letters if x and x in "ABCD"]
    if not valid:
        return ""
    cnt = Counter(valid)
    best_count = cnt.most_common(1)[0][1]
    winners = [c for c, n in cnt.items() if n == best_count]
    if len(winners) == 1:
        return winners[0]
    # 동점: 첫 번째 에이전트가 낸 답 중 하나를 선택 (에이전트 순서대로 먼저 나온 것)
    for L in letters:
        if L in winners:
            return L
    return winners[0]


def get_runner(model_name: str, config: dict):
    """run_eval.py와 동일한 runner 생성."""
    if model_name == "qwen":
        m_cfg = config.get("models", {}).get("qwen", {})
        if not m_cfg.get("enabled", True):
            return None
        return QwenRunner(
            model_id=m_cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "llava":
        m_cfg = config.get("models", {}).get("llava", {})
        if not m_cfg.get("enabled", True):
            return None
        return LLaVARunner(
            model_id=m_cfg.get("model_id", "llava-hf/llava-1.5-7b-hf"),
            device=m_cfg.get("device", "cuda"),
        )
    elif model_name == "gpt":
        m_cfg = config.get("models", {}).get("gpt", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            print(f"[skip] {model_name}: no OPENAI_API_KEY")
            return None
        return GPTRunner(
            model_id=m_cfg.get("model_id", "gpt-4o"),
            api_key=api_key,
        )
    elif model_name == "gemini":
        m_cfg = config.get("models", {}).get("gemini", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(f"[skip] {model_name}: no GEMINI_API_KEY")
            return None
        return GeminiRunner(
            model_id=m_cfg.get("model_id", "gemini-2.0-flash"),
            api_key=api_key,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_model_multiagent(
    model_name: str,
    dataset,
    config: dict,
    run_dir: Path,
) -> dict | None:
    """동일 모델 3명이 각자 답 → 다수결. 대화 로그 저장."""
    runner = get_runner(model_name, config)
    if runner is None:
        return None

    eval_cfg = config.get("eval", {})
    # 멀티에이전트: multi_agent_temperature 가 있으면 그걸 사용 (3명이 서로 다르게 답하게). 없으면 temperature (0이면 3명 답 동일)
    temp = eval_cfg.get("multi_agent_temperature")
    if temp is None:
        temp = eval_cfg.get("temperature", 0.0)
    if temp > 0:
        print(f"  [{model_name}] multi-agent temperature={temp} (3 agents can disagree)")
    max_new = eval_cfg.get("max_new_tokens", 512)
    top_k = eval_cfg.get("top_k", 0)
    top_p = eval_cfg.get("top_p", 0.0)
    use_max_tokens = model_name in ("gpt", "gemini")

    n = len(dataset)
    prompts = [get_prompt(dataset[i]) for i in range(n)]
    images = [dataset[i].get("images") or dataset[i].get("image") for i in range(n)]
    gt_list = [dataset[i]["answer_only"] for i in range(n)]

    preds_majority = []
    all_agent_letters = []  # list of [a1, a2, a3] per sample
    all_agent_raw = []      # list of [raw1, raw2, raw3] for conversation log

    for i in tqdm(range(n), desc=f"{model_name}(x{NUM_AGENTS})"):
        img = images[i]
        prompt = prompts[i]
        letters = []
        raws = []
        for _ in range(NUM_AGENTS):
            if use_max_tokens:
                out = runner.generate(img, prompt, temperature=temp, max_tokens=max_new, top_k=top_k, top_p=top_p)
            else:
                out = runner.generate(img, prompt, temperature=temp, max_new_tokens=max_new, top_k=top_k, top_p=top_p)
            raws.append(out)
            letters.append(normalize_answer_only(out))
        majority = majority_vote(letters)
        preds_majority.append(majority)
        all_agent_letters.append(letters)
        all_agent_raw.append(raws)

    acc = accuracy(preds_majority, gt_list)
    results = {
        "model": model_name,
        "accuracy": acc,
        "num_samples": n,
        "num_agents": NUM_AGENTS,
    }

    # 예측 저장 (jsonl)
    pred_path = run_dir / f"{model_name}_multiagent_preds.jsonl"
    with open(pred_path, "w") as f:
        for i in range(n):
            row = dataset[i]
            rec = {
                "idx": i,
                "question_only": row.get("question_only"),
                "category": row.get("category"),
                "agent1": all_agent_letters[i][0],
                "agent2": all_agent_letters[i][1],
                "agent3": all_agent_letters[i][2],
                "majority": preds_majority[i],
                "gt": gt_list[i],
                "correct": preds_majority[i] == gt_list[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 대화 로그 (텍스트): 질문, 3명 답, 다수결, 정답
    conv_dir = run_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    conv_path = conv_dir / f"{model_name}_conversations.txt"
    lines = [
        f"# Multi-agent conversations: {model_name} ({NUM_AGENTS} agents, majority vote)",
        f"# Total samples: {n}",
        "",
    ]
    for i in range(n):
        row = dataset[i]
        q = (row.get("question_only") or row.get("question_with_options") or "").strip()
        opts = row.get("options") or []
        opt_text = " / ".join([f"({chr(65+j)}) {o}" for j, o in enumerate(opts)])
        a1, a2, a3 = all_agent_letters[i]
        maj = preds_majority[i]
        gt = gt_list[i]
        correct = "✓" if maj == gt else "✗"
        lines.append(f"--- Sample {i} [{row.get('category', '')}] {correct} ---")
        lines.append(f"Q: {q}")
        if opt_text:
            lines.append(f"Options: {opt_text}")
        lines.append(f"Agent 1: {a1}  |  Agent 2: {a2}  |  Agent 3: {a3}")
        lines.append(f"→ Majority: {maj}  |  GT: {gt}")
        lines.append("")
    conv_path.write_text("\n".join(lines), encoding="utf-8")

    # 카테고리별 정확도 (단일 에이전트와 동일 형식)
    if "category" in dataset.features:
        by_cat = {}
        for i in range(n):
            c = dataset[i]["category"]
            if c not in by_cat:
                by_cat[c] = {"pred": [], "gt": []}
            by_cat[c]["pred"].append(preds_majority[i])
            by_cat[c]["gt"].append(gt_list[i])
        results["by_category"] = {c: accuracy(d["pred"], d["gt"]) for c, d in by_cat.items()}

    with open(run_dir / f"{model_name}_multiagent_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if results.get("by_category"):
        with open(run_dir / f"{model_name}_multiagent_by_category.json", "w") as f:
            json.dump(results["by_category"], f, indent=2)

    return results


def load_baseline_accuracies(baseline_run_dir: Path, models: list) -> dict:
    """단일 에이전트 run에서 모델별 정확도 로드."""
    accs = {}
    for model_name in models:
        # summary.json 또는 {model}_results.json
        p = baseline_run_dir / f"{model_name}_results.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            accs[model_name] = d.get("accuracy")
        else:
            summary_path = baseline_run_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    arr = json.load(f)
                for r in arr:
                    if r.get("model") == model_name:
                        accs[model_name] = r.get("accuracy")
                        break
    return accs


def main():
    parser = argparse.ArgumentParser(description="Multi-agent 평가 (3 agents, majority vote)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=["qwen", "llava"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_category", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--baseline_run_dir", type=str, default=None,
                        help="단일 에이전트 run 디렉터리. 지정 시 single vs multi 비교 결과 저장.")
    args = parser.parse_args()

    if any(m in args.models for m in ("qwen", "llava")):
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"[GPU] {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[WARN] CUDA not available.")

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    if args.max_samples is not None:
        ds_cfg = {**ds_cfg, "max_samples": args.max_samples}
    max_per_cat = args.max_per_category or ds_cfg.get("max_per_category")

    dataset = load_stvqa(
        dataset_name=ds_cfg.get("name", "OX-PIXL/STVQA-7K"),
        split=args.split,
        max_samples=ds_cfg.get("max_samples"),
        max_per_category=max_per_cat,
    )
    suffix = f", max_per_category={max_per_cat})" if max_per_cat else ""
    print(f"Loaded {len(dataset)} samples (split={args.split}{suffix})")
    print(f"Multi-agent: {NUM_AGENTS} agents per model, majority vote")

    output_dir = Path(args.output_dir or config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_multiagent"
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

    all_results = []
    for model_name in args.models:
        try:
            res = run_model_multiagent(model_name, dataset, config, run_dir)
            if res is not None:
                all_results.append(res)
                print(f"{model_name} (x{NUM_AGENTS} majority): accuracy = {res['accuracy']:.4f} ({res['num_samples']} samples)")
        except Exception as e:
            print(f"{model_name}: error - {e}")
            raise

    with open(run_dir / "summary_multiagent.json", "w") as f:
        json.dump(
            [{"model": r["model"], "accuracy": r["accuracy"], "num_samples": r["num_samples"], "num_agents": NUM_AGENTS}
            for r in all_results
        ],
        f,
        indent=2,
    )

    # 단일 vs 다중 비교
    if args.baseline_run_dir:
        baseline_dir = Path(args.baseline_run_dir).resolve()
        if baseline_dir.is_dir():
            single_acc = load_baseline_accuracies(baseline_dir, args.models)
            comparison = []
            for r in all_results:
                m = r["model"]
                multi_acc = r["accuracy"]
                s_acc = single_acc.get(m)
                delta = (multi_acc - s_acc) if s_acc is not None else None
                comparison.append({
                    "model": m,
                    "single_agent_accuracy": s_acc,
                    "multi_agent_accuracy": multi_acc,
                    "delta": round(delta, 4) if delta is not None else None,
                })
            with open(run_dir / "comparison_single_vs_multi.json", "w") as f:
                json.dump(comparison, f, indent=2)
            # 텍스트 요약
            comp_path = run_dir / "comparison_single_vs_multi.txt"
            comp_lines = [
                "Single vs Multi-agent (3 agents, majority vote)",
                f"Baseline (single): {baseline_dir}",
                "",
                "model          | single (1 agent) | multi (3 agents) | delta",
                "-" * 60,
            ]
            for c in comparison:
                s = f"{c['single_agent_accuracy']:.2%}" if c["single_agent_accuracy"] is not None else "N/A"
                mu = f"{c['multi_agent_accuracy']:.2%}"
                d = f"{c['delta']:+.2%}" if c["delta"] is not None else "N/A"
                comp_lines.append(f"{c['model']:14} | {s:16} | {mu:16} | {d}")
            comp_path.write_text("\n".join(comp_lines), encoding="utf-8")
            csv_path = run_dir / "comparison_single_vs_multi.csv"
            with open(csv_path, "w") as f:
                f.write("model,single_agent_accuracy,multi_agent_accuracy,delta\n")
                for c in comparison:
                    s = f"{c['single_agent_accuracy']:.4f}" if c["single_agent_accuracy"] is not None else ""
                    mu = f"{c['multi_agent_accuracy']:.4f}"
                    d = f"{c['delta']:.4f}" if c["delta"] is not None else ""
                    f.write(f"{c['model']},{s},{mu},{d}\n")
            print(f"Comparison saved: {comp_path}, {csv_path}")
        else:
            print(f"[WARN] Baseline run dir not found: {baseline_dir}")

    print(f"Results saved to {run_dir}")
    print(f"Conversations: {run_dir / 'conversations'}")


if __name__ == "__main__":
    main()
