"""
Phase 1: GT와 비교해서 모델별 보상 주기.

- sim_i = sim(a_i, y) ∈ [0,1]
- Soft: R_i = 2·sim_i - 1
- Case 1 (최종 정답): R_i 그대로
- Case 2 (최종 오답): R_i = 2·sim_i - 1 - κ·δ_i
  δ_i = max(0, sim_final - sim_i)
"""
import re
from typing import Callable, Dict, List, Optional

# Default: multiple choice (A/B/C/D) exact match
def _normalize_mc(s: str) -> str:
    """Extract (A)/(B)/(C)/(D) from answer string."""
    s = (s or "").strip().upper()
    m = re.search(r"\(([A-D])\)", s)
    if m:
        return m.group(1)
    m = re.search(r"(?:ANSWER|FINAL\s*ANSWER)[:\s]+([A-D])\b", s, re.I)
    if m:
        return m.group(1)
    tail = s[-400:] if len(s) > 400 else s
    all_m = re.findall(r"\b([A-D])\b", tail)
    return all_m[-1] if all_m else ""


def similarity_answer(
    pred: str,
    gt: str,
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> float:
    """
    sim(a_i, y) ∈ [0,1].
    Multiple choice: 1.0 if match else 0.0.
    """
    norm = normalize_fn or _normalize_mc
    p = norm(pred)
    g = norm(gt)
    if not g:
        return 0.5  # No GT: neutral
    return 1.0 if p == g else 0.0


def compute_rewards(
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    kappa: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, float]:
    """
    R_i for each agent.

    Args:
        agent_answers: {agent_id: a_i} (각 agent 출력)
        final_answer: ŷ (Reasoning agent 최종 답)
        gt_answer: y (Ground Truth)
        kappa: Case 2 추가 패널티 강도

    Returns:
        {agent_id: R_i}
    """
    sim_fn = similarity_fn or similarity_answer
    final_correct = sim_fn(final_answer, gt_answer) >= 0.99  # Case 1 vs 2
    sim_final = sim_fn(final_answer, gt_answer)

    rewards = {}
    for agent_id, a_i in agent_answers.items():
        sim_i = sim_fn(a_i, gt_answer)
        # Soft: R_i = 2·sim_i - 1
        R_i = 2.0 * sim_i - 1.0

        if not final_correct:
            # Case 2: δ_i = max(0, sim_final - sim_i)
            delta_i = max(0.0, sim_final - sim_i)
            R_i = R_i - kappa * delta_i

        rewards[agent_id] = R_i

    return rewards
