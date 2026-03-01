"""
Confidence-based ScoreMap: run_step1 업데이트 + select_agents_by_score 선택.

- step=0: 고정 qwen3_4b (3 roles 모두)
- step>0: confidence_score.select_agents_by_score로 선택
- 업데이트: run_step1 (reward 기반)
"""
import copy
from typing import Dict, List, Optional, Tuple

from .config import ALL_CATEGORIES, SPECIALIST_LLMS, ROLES
from .score_map import ScoreMap

# spatial_aomas
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from spatial_aomas.confidence_score import (
    select_agents_by_score,
    run_step1,
    ROLES_MASV2,
)

FIXED_FIRST_LLM = "qwen3_4b"


def _score_map_to_scores(score_map: ScoreMap) -> Dict[str, Dict[str, Dict[str, float]]]:
    """ScoreMap._maps[cat][role][llm] → scores[llm][cat][role]"""
    scores = {}
    for llm in score_map.llms:
        scores[llm] = {}
        for cat in score_map.categories:
            scores[llm][cat] = {}
            for role in score_map.roles:
                v = score_map.get_score(cat, role, llm)
                scores[llm][cat][role] = v
    return scores


def _scores_to_score_map(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    score_map: ScoreMap,
) -> None:
    """scores[llm][cat][role] → ScoreMap._maps (in-place)"""
    for llm, cats in scores.items():
        for cat, roles in cats.items():
            for role, v in roles.items():
                if cat in score_map._maps and role in score_map._maps[cat]:
                    score_map.set_score(cat, role, llm, max(0.0, min(1.0, v)))


class ConfidenceScoreMapUpdater:
    """run_step1 기반 업데이터. ScoreMapUpdater 인터페이스 호환."""

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

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
        """run_step1으로 점수 갱신."""
        agent_answers = {r["llm_name"]: r.get("answer", "") for r in agent_results}
        agent_roles = {llm: role for role, llm in assignments}

        scores = _score_map_to_scores(score_map)
        updated = run_step1(
            scores=scores,
            agent_answers=agent_answers,
            final_answer=final_answer,
            gt_answer=gt,
            category=category,
            agent_roles=agent_roles,
            kappa=self.kappa,
        )
        _scores_to_score_map(updated, score_map)


class ConfidenceScoreMap(ScoreMap):
    """step=0: 고정 qwen3_4b. step>0: select_agents_by_score."""

    def select_agents(
        self, category: str, step: int,
    ) -> List[Tuple[str, str]]:
        """step=0: qwen3_4b 고정. step>0: confidence 기반 선택."""
        if step == 0:
            return [(role, FIXED_FIRST_LLM) for role in self.roles]

        scores = _score_map_to_scores(self)
        assignment = select_agents_by_score(
            scores, category, self.llms, roles=ROLES_MASV2,
        )
        return [(role, assignment.get(role, self.llms[0])) for role in self.roles]
