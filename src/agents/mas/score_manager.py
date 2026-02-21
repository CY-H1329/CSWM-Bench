"""
Score table: s[agent, role, category] for agent selection and role assignment.

Initial: 0.5. Correct: +0.05. Wrong: -0.02.
Role assignment: i_k* = argmax_i s[i,k,c] (score-based, not agent self-selection).
"""
from typing import Dict, List, Optional, Tuple

from .config import (
    CANDIDATE_AGENTS,
    TASK_CATEGORIES,
    MAS_ROLES,
    INITIAL_WEIGHT,
    SCORE_DELTA_CORRECT,
    SCORE_DELTA_WRONG,
)


class ScoreManager:
    """
    Manages per-agent per-role per-category scores: s[agent, role, category].

    Role assignment is score-based: for role k and category c,
    assign to agent i_k* = argmax_i s[i,k,c].
    """

    def __init__(self):
        # _scores[agent][category][role] = float
        self._scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        for m in CANDIDATE_AGENTS:
            self._scores[m] = {}
            for c in TASK_CATEGORIES:
                self._scores[m][c] = {r: INITIAL_WEIGHT for r in MAS_ROLES}

    def get(self, agent: str, category: str, role: Optional[str] = None) -> float:
        """
        Get score for (agent, category). If role given, return s[agent, role, category].
        If role is None, return mean over roles (agent-level for category).
        """
        if agent not in self._scores or category not in self._scores[agent]:
            return INITIAL_WEIGHT
        role_scores = self._scores[agent][category]
        if role is not None:
            return role_scores.get(role, INITIAL_WEIGHT)
        return sum(role_scores.values()) / len(MAS_ROLES) if role_scores else INITIAL_WEIGHT

    def get_for_role(self, agent: str, role: str, category: str) -> float:
        """Get s[agent, role, category] — for role assignment."""
        return self.get(agent, category, role)

    def update(
        self,
        agent: str,
        category: str,
        correct: bool,
        role: Optional[str] = None,
    ):
        """
        Update score. If role given, update only that role.
        If role is None, update all roles (legacy behavior).
        """
        if agent not in self._scores:
            self._scores[agent] = {c: {r: INITIAL_WEIGHT for r in MAS_ROLES} for c in TASK_CATEGORIES}
        if category not in self._scores[agent]:
            self._scores[agent][category] = {r: INITIAL_WEIGHT for r in MAS_ROLES}

        delta = SCORE_DELTA_CORRECT if correct else SCORE_DELTA_WRONG
        roles_to_update = [role] if role and role in MAS_ROLES else MAS_ROLES
        for r in roles_to_update:
            old = self._scores[agent][category].get(r, INITIAL_WEIGHT)
            self._scores[agent][category][r] = max(0.0, min(1.0, old + delta))

    def get_top_k(self, category: str, k: int = 3, role: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Return top-k agents by score for this category.
        If role given, use s[agent, role, category]; else use mean over roles.
        """
        pairs = []
        for m in CANDIDATE_AGENTS:
            s = self.get(m, category, role)
            pairs.append((m, s))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def assign_roles(
        self,
        category: str,
        candidate_agents: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Assign each role to one agent (1:1) by score.
        Returns {role: agent} for role in MAS_ROLES.
        Uses linear assignment to maximize sum of s[agent, role, category].
        """
        agents = list(candidate_agents or CANDIDATE_AGENTS)
        roles = list(MAS_ROLES)

        if len(agents) < len(roles):
            roles = roles[: len(agents)]
        if len(roles) == 0:
            return {}

        # Greedy 1:1 assignment: for each role, pick best unused agent
        assignment: Dict[str, str] = {}
        used_agents: set = set()
        for role in roles:
            best_agent = None
            best_score = -1.0
            for agent in agents:
                if agent in used_agents:
                    continue
                s = self.get_for_role(agent, role, category)
                if s > best_score:
                    best_score = s
                    best_agent = agent
            if best_agent is not None:
                assignment[role] = best_agent
                used_agents.add(best_agent)
            elif agents:
                assignment[role] = agents[0]
        return assignment

    def get_role_scores_table(self, category: str) -> Dict[str, Dict[str, float]]:
        """
        Return {agent: {role: score}} for this category.
        Used to load scores for pipeline / Head output.
        """
        out = {}
        for agent in CANDIDATE_AGENTS:
            out[agent] = {
                role: self.get_for_role(agent, role, category)
                for role in MAS_ROLES
            }
        return out

    def to_dict(self) -> dict:
        """Serialize for persistence (full 3D: agent -> category -> role -> score)."""
        return {
            m: {
                c: dict(self._scores[m][c])
                for c in self._scores[m]
            }
            for m in self._scores
        }

    def to_dict_flat(self) -> Dict[str, Dict[str, float]]:
        """Flatten for display: agent -> category -> mean(role scores)."""
        out = {}
        for m in self._scores:
            out[m] = {}
            for c in self._scores[m]:
                roles = self._scores[m][c]
                out[m][c] = sum(roles.values()) / len(roles) if roles else INITIAL_WEIGHT
        return out

    def from_dict(self, data: dict):
        """Load from persisted dict. Supports 3D (role scores) or 2D (flat) format."""
        for m, cats in data.items():
            if m not in self._scores:
                self._scores[m] = {}
            for c, val in cats.items():
                if isinstance(val, dict):
                    self._scores[m][c] = {r: float(v) for r, v in val.items() if r in MAS_ROLES}
                else:
                    # Legacy: single float -> same score for all roles
                    self._scores[m][c] = {r: float(val) for r in MAS_ROLES}
