#!/usr/bin/env python3
"""
MAS v2 — 전문가 LLM 고정 테스트 (trust/confidence 없음).

Head + Final Reasoner는 동일. 중간 3 roles 모두 동일한 specialist 모델 고정.
- sa2va만 사용, qwen3_4b만 사용 등 선택 가능
- 스코어 업데이트 없음 (update_scores=False)

Usage (CLI):
    python test_fixed_specialist_mas_v2.py --specialist sa2va --benchmark cvbench --max_samples 50

Usage (Jupyter):
    import sys
    sys.path.insert(0, "/path/to/Spatial_MAS")  # 프로젝트 경로

    from test_fixed_specialist_mas_v2 import run_fixed_specialist_mas_test, build_runners_fixed

    head_gen, spec_gen, reason_gen = build_runners_fixed(
        specialist="sa2va",
        specialist_device="cuda",
        use_vlm_reasoning=True,
    )
    results = run_fixed_specialist_mas_test(
        head_gen, spec_gen, reason_gen,
        specialist="sa2va",
        benchmark="cvbench",
        max_samples=50,
    )
    print(f"Accuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src2.agents.mas_v2 import ALL_CATEGORIES, ROLES, run_step
from src2.agents.mas_v2.score_map import ScoreMap

from src2.benchmarks.loaders import (
    load_benchmark,
    get_benchmark_image,
    get_benchmark_prompt,
    get_benchmark_answer,
)


class FixedSpecialistScoreMap(ScoreMap):
    """모든 role에 동일한 specialist 고정. trust/confidence 없음."""

    def __init__(self, fixed_model: str, categories: List[str], roles: Optional[List[str]] = None):
        super().__init__(categories=categories, llms=[fixed_model], roles=roles or ROLES)
        self.fixed_model = fixed_model

    def select_agents(self, category: str, step: int) -> List[Tuple[str, str]]:
        return [(role, self.fixed_model) for role in self.roles]


def build_runners_fixed(
    specialist: str = "sa2va",
    specialist_device: str = "cuda",
    use_vlm_reasoning: bool = True,
):
    """build_runners + specialist_whitelist=[specialist] (해당 모델만 로드)."""
    from run_eval_mas_v2 import build_runners
    return build_runners(
        specialist_whitelist=[specialist],
        specialist_device=specialist_device,
        use_vlm_reasoning=use_vlm_reasoning,
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


def run_fixed_specialist_mas_test(
    head_generate,
    specialist_generate,
    reasoning_generate,
    specialist: str = "sa2va",
    benchmark: str = "cvbench",
    max_samples: int = 50,
    seed: int = 42,
    use_vlm_reasoning: bool = True,
):
    """
    전문가 LLM 고정 MAS v2 실행 (trust 없음).
    Returns: accuracy, correct, total, per_category
    """
    dataset = load_benchmark(benchmark, max_samples=max_samples, seed=seed)
    samples = []
    for i in range(len(dataset)):
        r = _prefetch_sample(dataset[i], benchmark, i)
        if r is not None:
            samples.append(r)

    rng = random.Random(seed)
    rng.shuffle(samples)

    score_map = FixedSpecialistScoreMap(fixed_model=specialist, categories=ALL_CATEGORIES)

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"Fixed Specialist MAS v2 — {benchmark.upper()} (n={len(samples)})")
    print(f"  specialist: {specialist} (3 roles 모두 고정)")
    print(f"  trust/confidence: 없음")
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
            updater=None,
            update_scores=False,
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

        acc = correct / total if total > 0 else 0
        print(f"  Step {step+1}/{len(samples)} | acc: {100*acc:.1f}% | cat: {cat} | "
              f"assign: {[(r, l) for r, l in result.get('assignments', [])]}")

    # 최종 리포트
    print()
    print("=" * 70)
    print(f"FIXED SPECIALIST MAS v2 — {benchmark.upper()} — 최종 결과")
    print("=" * 70)
    print(f"Specialist: {specialist} (3 roles 고정)")
    print(f"Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print()
    for cat in sorted(by_category.keys()):
        v = by_category[cat]
        if v["total"] > 0:
            acc = v["correct"] / v["total"]
            print(f"  {cat:35s}  {100*acc:5.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 70)
    print()
    print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    return {
        "benchmark": benchmark,
        "specialist": specialist,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_category": dict(by_category),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", default="sa2va", help="고정 specialist (sa2va, qwen3_4b, llava4d, spatial_reasoner, spatial_rgpt)")
    parser.add_argument("--benchmark", default="cvbench", choices=["cvbench", "3dsrbench"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    head_gen, spec_gen, reason_gen = build_runners_fixed(
        specialist=args.specialist,
        specialist_device=args.device,
        use_vlm_reasoning=True,
    )

    results = run_fixed_specialist_mas_test(
        head_gen, spec_gen, reason_gen,
        specialist=args.specialist,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        seed=args.seed,
    )
