"""
Score Map Updater -- interface + default implementation.

INPUT (fixed interface):
    score_map   : ScoreMap instance
    category    : str -- inferred category for this step
    assignments : [(role, llm), ...] -- the 3 role-LLM pairs used
    agent_results : [{"role", "llm_name", "answer", "reason"}, ...]
    final_answer: str -- the Final Reasoning Agent's answer
    gt          : str -- ground-truth answer
    step        : int -- current step index (0-based)
    total_steps : int -- total number of steps

OUTPUT:
    The score_map is mutated in-place (relevant cells updated).

Override `update()` to change the scoring logic.
"""
import re
from typing import Dict, List, Tuple

from .score_map import ScoreMap
from .config import DEFAULT_SCORE_DELTA_CORRECT, DEFAULT_SCORE_DELTA_WRONG


def _normalise_answer(raw: str) -> str:
    """Extract answer letter from raw text: '(A)', 'A', 'Answer: B' -> 'A'."""
    raw = (raw or "").strip().upper()
    m = re.search(r"\(([A-D])\)", raw)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", raw)
    if m:
        return m.group(1)
    return raw


class ScoreMapUpdater:
    """Default updater: +alpha if agent correct, -beta if wrong (per cell)."""

    def __init__(
        self,
        alpha: float = DEFAULT_SCORE_DELTA_CORRECT,
        beta: float = abs(DEFAULT_SCORE_DELTA_WRONG),
    ):
        self.alpha = alpha
        self.beta = beta

    def update(
        self,
        score_map: ScoreMap,
        category: str,
        assignments: List[Tuple[str, str]],
        agent_results: List[Dict],
        final_answer: str,
        gt: str,
        step: int,
        total_steps: int,
    ) -> None:
        """Update score_map in-place.

        Default: compare each agent's individual answer to GT.
        Correct -> cell += alpha.  Wrong -> cell -= beta.
        """
        gt_norm = _normalise_answer(gt)
        for (role, llm), result in zip(assignments, agent_results):
            pred_norm = _normalise_answer(result.get("answer", ""))
            if pred_norm == gt_norm:
                delta = self.alpha
            else:
                delta = -self.beta
            current = score_map.get_score(category, role, llm)
            score_map.set_score(category, role, llm, current + delta)
