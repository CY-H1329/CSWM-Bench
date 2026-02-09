"""
STVQA-7K dataset loader for evaluation.
Dataset: https://huggingface.co/datasets/OX-PIXL/STVQA-7K
Paper: SpatialThinker (arXiv:2511.07403)
"""
import random
from typing import Optional

from datasets import load_dataset


def load_stvqa(
    dataset_name: str = "OX-PIXL/STVQA-7K",
    split: str = "val",
    max_samples: Optional[int] = None,
    max_per_category: Optional[int] = None,
    seed: int = 42,
):
    """
    Load STVQA-7K for evaluation.
    - split: "val" (692) or "train" (6895)
    - max_samples: 전체 상한 (앞에서부터)
    - max_per_category: 카테고리별 최대 개수 (균등 샘플링). 모든 task를 골고루 실험할 때 사용.
    """
    dataset = load_dataset(dataset_name, split=split)
    if max_per_category is not None and "category" in dataset.features:
        rng = random.Random(seed)
        by_cat = {}
        for i in range(len(dataset)):
            c = dataset[i].get("category") or "unknown"
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(i)
        indices = []
        for c in sorted(by_cat.keys()):
            idx_list = by_cat[c]
            k = min(max_per_category, len(idx_list))
            indices.extend(rng.sample(idx_list, k))
            if max_per_category < len(idx_list):
                print(f"  [data] {c}: {k}/{len(idx_list)} (capped by max_per_category)")
            else:
                print(f"  [data] {c}: {k}/{len(idx_list)}")
        indices.sort()
        dataset = dataset.select(indices)
    elif max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def get_prompt(example: dict, include_options: bool = True) -> str:
    """Build prompt string for a sample (question + options)."""
    q = example.get("question_only") or example.get("question_with_options") or ""
    if not include_options:
        return q
    opts = example.get("options", [])
    if opts:
        lines = [q, "Options:"]
        for i, o in enumerate(opts):
            label = chr(65 + i)  # A, B, C, D
            lines.append(f"({label}) {o}")
        return "\n".join(lines)
    return q


def normalize_answer_only(pred: str) -> str:
    """Extract answer letter (A/B/C/D) from model output for matching."""
    pred = (pred or "").strip().upper()
    for c in "ABCD":
        if c in pred:
            # take first occurrence as answer choice
            idx = pred.index(c)
            # accept "(A)" or "A)" or "A."
            rest = pred[idx : idx + 4]
            if rest.startswith("(") and len(rest) >= 2:
                return rest[1]
            return c
    return ""


def accuracy(pred_letters: list, gt_letters: list) -> float:
    """Accuracy from list of predicted and ground-truth answer letters."""
    assert len(pred_letters) == len(gt_letters)
    if not pred_letters:
        return 0.0
    return sum(p == g for p, g in zip(pred_letters, gt_letters)) / len(pred_letters)
