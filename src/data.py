"""
STVQA-7K dataset loader for evaluation.
Dataset: https://huggingface.co/datasets/OX-PIXL/STVQA-7K
Paper: SpatialThinker (arXiv:2511.07403)
"""
import random
import re
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


def get_prompt_with_reasoning(example: dict) -> str:
    """협의용: 질문+옵션 + 추론 한 문장 후 (A)/(B)/(C)/(D) 로 답하라고 안내."""
    base = get_prompt(example, include_options=True)
    return (
        base
        + "\n\nReply in this format:\nReasoning: [one sentence explaining why]\nAnswer: (A) or (B) or (C) or (D)"
    )


def normalize_answer_only(pred: str) -> str:
    """Extract answer letter (A/B/C/D) from model output for matching."""
    pred = (pred or "").strip().upper()
    # 1) "(A)", "(B)", "(C)", "(D)" — évite "ANSWER" qui contient A
    matches = re.findall(r"\(([A-D])\)", pred)
    if matches:
        return matches[-1]
    # 2) "Answer: A", "Final Answer: B" etc.
    m = re.search(r"(?:ANSWER|FINAL\s*ANSWER)[:\s]+([A-D])\b", pred, re.I)
    if m:
        return m.group(1).upper()
    # 3) Dernier A/B/C/D isolé (word boundary)
    all_matches = re.findall(r"\b([A-D])\b", pred)
    return all_matches[-1] if all_matches else ""


def accuracy(pred_letters: list, gt_letters: list) -> float:
    """Accuracy from list of predicted and ground-truth answer letters."""
    assert len(pred_letters) == len(gt_letters)
    if not pred_letters:
        return 0.0
    return sum(p == g for p, g in zip(pred_letters, gt_letters)) / len(pred_letters)


def normalize_category(cat: str) -> str:
    """Normalize 3DSRBench category for comparison."""
    if not cat or not str(cat).strip():
        return ""
    s = str(cat).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("multi_object", "multiobject"):
        return "Multi-Object"
    if s == "height":
        return "Height"
    if s == "location":
        return "Location"
    if s == "orientation":
        return "Orientation"
    return cat.strip()  # keep original if unknown


def extract_predicted_category(response: str) -> str:
    """Extract Task Category from model output (STEP 1 classification)."""
    if not response or not response.strip():
        return ""
    text = response.strip()
    # Match "Task Category:" or "Task Category" followed by category on same or next line
    m = re.search(
        r"Task\s*Category\s*:?\s*\n?\s*([A-Za-z][A-Za-z\s\-_]*?)(?=\n\n|\nReasoning|\nStep-by-Step|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip()
        return normalize_category(raw) if raw else ""
    # Fallback: first line that is exactly one of the 4 categories
    for line in text.split("\n"):
        line = line.strip()
        if normalize_category(line) in ("Height", "Location", "Orientation", "Multi-Object"):
            return normalize_category(line)
    return ""
