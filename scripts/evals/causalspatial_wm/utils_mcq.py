from __future__ import annotations

import re
from typing import Dict, List, Tuple


def build_mcq_prompt(question: str, choices: Dict[str, str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for k in ("A", "B", "C", "D"):
        if k in choices:
            lines.append(f"({k}) {choices[k]}")
    lines.append("")
    lines.append("Return ONLY the letter: A/B/C/D.")
    return "\n".join(lines)


def extract_letter(text: str) -> str:
    if not text:
        return ""
    s = text.strip().upper()
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)
    # common patterns like "(C)" or "Answer: C"
    m2 = re.search(r"ANSWER[^A-D]*([A-D])", s)
    if m2:
        return m2.group(1)
    return s[:1] if s[:1] in "ABCD" else ""


def accuracy(preds: List[str], gts: List[str]) -> float:
    if not preds:
        return 0.0
    n = min(len(preds), len(gts))
    if n == 0:
        return 0.0
    c = sum(1 for i in range(n) if preds[i] == gts[i])
    return c / n

