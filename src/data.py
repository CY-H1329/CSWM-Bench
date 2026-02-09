"""
STVQA-7K dataset loader for evaluation.
Dataset: https://huggingface.co/datasets/OX-PIXL/STVQA-7K
Paper: SpatialThinker (arXiv:2511.07403)
"""
from typing import Optional

from datasets import load_dataset


def load_stvqa(
    dataset_name: str = "OX-PIXL/STVQA-7K",
    split: str = "val",
    max_samples: Optional[int] = None,
):
    """
    Load STVQA-7K for evaluation.
    - split: "val" (692) or "train" (6895)
    - Each sample: image, question_only, options, answer_only (A/B/C/D), category, etc.
    """
    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
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
