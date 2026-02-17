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
    # 1) "(A)", "(B)", "(C)", "(D)"
    matches = re.findall(r"\(([A-D])\)", pred)
    if matches:
        return matches[-1]
    # 2) "Answer: A", "Final Answer: B", "The answer is A", "answer is (B)"
    m = re.search(r"(?:ANSWER|FINAL\s*ANSWER)[:\s]+([A-D])\b", pred, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s+(?:\(?([A-D])\)?|OPTION\s+([A-D]))", pred, re.I)
    if m:
        return (m.group(1) or m.group(2)).upper()
    m = re.search(r"OPTION\s+([A-D])\b|CHOICE\s+([A-D])\b|SELECT\s+([A-D])\b", pred, re.I)
    if m:
        return (m.group(1) or m.group(2) or m.group(3)).upper()
    m = re.search(r"(?:I\s+)?(?:WOULD\s+)?(?:CHOOSE|SELECT)\s+(?:OPTION\s+)?\(?([A-D])\)?", pred, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"THEREFORE[,:]?\s*\(?([A-D])\)?\.?\s*$", pred, re.I | re.M)
    if m:
        return m.group(1).upper()
    # 3) Last A/B/C/D in final 400 chars (évite "A" dans "ANSWER" au début)
    tail = pred[-400:] if len(pred) > 400 else pred
    all_matches = re.findall(r"\b([A-D])\b", tail)
    return all_matches[-1] if all_matches else ""


def normalize_answer_freeform(pred: str) -> str:
    """Extract free-form answer from model output (for GQA, etc.). Returns normalized string."""
    pred = (pred or "").strip()
    # "Answer: X", "The answer is X", "Final Answer: X"
    m = re.search(r"(?:ANSWER|FINAL\s*ANSWER|THE\s+ANSWER\s+IS)[:\s]+(.+?)(?:\n|$)", pred, re.I | re.DOTALL)
    if m:
        return m.group(1).strip().lower()
    # Last non-empty line
    lines = [l.strip() for l in pred.split("\n") if l.strip()]
    return lines[-1].lower() if lines else ""


def accuracy(pred_letters: list, gt_letters: list) -> float:
    """Accuracy from list of predicted and ground-truth answer letters."""
    assert len(pred_letters) == len(gt_letters)
    if not pred_letters:
        return 0.0
    return sum(p == g for p, g in zip(pred_letters, gt_letters)) / len(pred_letters)


def accuracy_freeform(preds: list, gts: list) -> float:
    """Accuracy for free-form answers (normalized: lowercase, strip)."""
    assert len(preds) == len(gts)
    if not preds:
        return 0.0
    return sum(str(p).strip().lower() == str(g).strip().lower() for p, g in zip(preds, gts)) / len(preds)


# 3DSRBench fine-grained categories (for normalization)
_3DSRBENCH_CATS = frozenset({
    "location_above", "height_higher", "location_closer_to_camera",
    "multi_object_closer_to", "orientation_on_the_left", "multi_object_facing",
    "multi_object_same_direction", "orientation_in_front_of",
    "multi_object_viewpoint_towards_object", "orientation_viewpoint",
    "location_next_to", "multi_object_parallel",
})


def normalize_category(cat: str) -> str:
    """Normalize 3DSRBench category for comparison (12 fine-grained categories)."""
    if not cat or not str(cat).strip():
        return ""
    s = str(cat).strip().lower().replace(" ", "_").replace("-", "_")
    if s in _3DSRBENCH_CATS:
        return s
    return cat.strip()


def extract_predicted_category(response: str, valid_cats: Optional[set] = None) -> str:
    """Extract Task Category/Type from model output (STEP 1 classification)."""
    if not response or not response.strip():
        return ""
    text = response.strip()
    # Match "Task Category:" or "Task Type:" followed by category
    m = re.search(
        r"Task\s*(?:Category|Type)\s*:?\s*\n?\s*([A-Za-z][A-Za-z0-9_\s\-]*?)(?=\n\n|\nReasoning|\nStep-by-Step|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        cats = valid_cats or _3DSRBENCH_CATS
        return raw if raw in cats else raw
    # Fallback: first line that matches a valid category
    cats = valid_cats or _3DSRBENCH_CATS
    for line in text.split("\n"):
        line = line.strip().lower().replace(" ", "_").replace("-", "_")
        if line in cats:
            return line
    return ""
