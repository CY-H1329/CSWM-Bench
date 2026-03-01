#!/usr/bin/env python3
"""
Confidence-based MAS v2 테스트: run_step1 업데이트 + select_agents_by_score 선택.

- step=0: qwen3_4b 고정 (3 roles)
- step>0: confidence 기반 LLM 선택
- 벤치마크당 10개, 스코어 맵 히스토리 기록, 최종 정확도 출력

Usage:
    python test_confidence_mas_v2.py --benchmark cvbench --max_samples 10
    python test_confidence_mas_v2.py --benchmark 3dsrbench --max_samples 10
"""
import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import ALL_CATEGORIES, run_step
from src2.agents.mas_v2.confidence_score_map import (
    ConfidenceScoreMap,
    ConfidenceScoreMapUpdater,
)
from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
)


def _prefetch_sample(ex, benchmark, i):
    image = get_benchmark_image(ex, benchmark)
    if image is None:
        return None
    query = get_benchmark_prompt(ex, benchmark)
    gt_raw = get_benchmark_answer(ex, benchmark)
    gt = (gt_raw or "").strip().upper()
    if not any(c in gt for c in "ABCD"):
        return None
    return {"index": i, "image": image, "query": query, "gt": gt}


def run_confidence_mas_test(
    head_generate,
    specialist_generate,
    reasoning_generate,
    benchmark: str = "cvbench",
    max_samples: int = 10,
    seed: int = 42,
    use_vlm_reasoning: bool = False,
):
    """
    Confidence 기반 MAS v2 실행.
    Returns: accuracy, correct, total, score_history, final_map
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    samples = []
    for i in range(len(dataset)):
        r = _prefetch_sample(dataset[i], benchmark, i)
        if r is not None:
            samples.append(r)

    rng = random.Random(seed)
    rng.shuffle(samples)

    score_map = ConfidenceScoreMap(categories=ALL_CATEGORIES, seed=seed)
    updater = ConfidenceScoreMapUpdater()

    score_history = []
    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"Confidence MAS v2 — {benchmark.upper()} (n={len(samples)})")
    print(f"  step=0: qwen3_4b 고정, step>0: run_step1 기반 선택")
    print()

    for step, s in enumerate(samples):
        result = run_step(
            image=s["image"],
            query=s["query"],
            gt=s["gt"],
            step=step,
            total_steps=len(samples),
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=updater,
            update_scores=True,
            use_vlm_reasoning=use_vlm_reasoning,
        )

        hit = result.get("correct", False)
        total += 1
        if hit:
            correct += 1
        cat = result.get("category", "unknown")
        by_category[cat]["total"] += 1
        if hit:
            by_category[cat]["correct"] += 1

        # 스코어 맵 히스토리 기록 (deep copy)
        step_scores = copy.deepcopy(score_map.get_all_maps())
        score_history.append({
            "step": step,
            "category": cat,
            "assignments": result.get("assignments", []),
            "correct": hit,
            "scores": step_scores,
        })

        acc = correct / total if total > 0 else 0
        print(f"  Step {step+1}/{len(samples)} | acc: {100*acc:.1f}% | cat: {cat} | "
              f"assign: {[(r, l) for r, l in result.get('assignments', [])]}")
        print(f"    scores (step {step}): {json.dumps(step_scores, ensure_ascii=False)}")

    # 최종 리포트
    print()
    print("=" * 70)
    print(f"CONFIDENCE MAS v2 — {benchmark.upper()} — 최종 결과")
    print("=" * 70)
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print()
    for cat in sorted(by_category.keys()):
        v = by_category[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:35s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 70)

    # 최종 스코어 맵 출력
    print()
    print("최종 스코어 맵 (category별 role×llm):")
    final_map = score_map.get_all_maps()
    for cat in sorted(final_map.keys()):
        print(f"\n  [{cat}]")
        for role, llm_scores in sorted(final_map[cat].items()):
            top = max(llm_scores.items(), key=lambda x: x[1])
            print(f"    {role}: {top[0]}={top[1]:.3f}  (전체: {llm_scores})")
    print()

    return {
        "benchmark": benchmark,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_category": dict(by_category),
        "score_history": score_history,
        "final_map": final_map,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="cvbench", choices=["cvbench", "3dsrbench"])
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="", help="Save results JSON path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from run_eval_mas_v2 import build_runners
    head_gen, spec_gen, reason_gen = build_runners(specialist_device=args.device)

    results = run_confidence_mas_test(
        head_gen, spec_gen, reason_gen,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if args.output:
        # score_history의 scores는 직렬화 가능
        out = {
            "benchmark": results["benchmark"],
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "per_category": results["per_category"],
            "final_map": results["final_map"],
            "score_history": [
                {
                    "step": h["step"],
                    "category": h["category"],
                    "assignments": h["assignments"],
                    "correct": h["correct"],
                    "scores": h["scores"],
                }
                for h in results["score_history"]
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Saved to {args.output}")

    print(f"\nSUMMARY: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
