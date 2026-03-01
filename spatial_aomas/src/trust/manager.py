"""
TrustManager — 통합 인터페이스.

Pipeline에서 호출:
1. compute_rewards(agent_answers, final_answer, gt)
2. scale_rewards(rewards, N_c)
3. update_credibility_full(...) 또는 update_scores_simple(...)
"""
from typing import Dict, List, Optional

from .reward import compute_rewards, similarity_answer
from .scaling import scale_rewards, phi_scale, update_scores_simple
from .credibility import (
    update_credibility_full,
    TrustState,
    trust_state_to_scores,
)


class TrustManager:
    """
    Trust Score 통합 관리.
    - N_c per category (누적 샘플 수)
    - Phase 선택: "simple" (Phase 2) vs "full" (Phase 3)
    """

    def __init__(
        self,
        agents: List[str],
        categories: List[str],
        roles: List[str],
        phase: str = "full",
        T: float = 10.0,
        gamma: float = 0.1,
        kappa: float = 1.0,
        lambda_f: float = 0.3,
        lambda_g: float = 0.1,
        mu: float = 0.5,
    ):
        self.agents = agents
        self.categories = categories
        self.roles = roles
        self.phase = phase
        self.T = T
        self.gamma = gamma
        self.kappa = kappa
        self.lambda_f = lambda_f
        self.lambda_g = lambda_g
        self.mu = mu

        self.N_c: Dict[str, int] = {c: 0 for c in categories}

        if phase == "full":
            self._trust_state: Dict[str, Dict[str, Dict[str, TrustState]] = {}
            for a in agents:
                self._trust_state[a] = {}
                for c in categories:
                    self._trust_state[a][c] = {
                        r: TrustState() for r in roles
                    }
            self._scores: Optional[Dict] = None
        else:
            self._scores = {}
            for a in agents:
                self._scores[a] = {}
                for c in categories:
                    self._scores[a][c] = {r: 0.5 for r in roles}
            self._trust_state = None

    def get_scores(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """s[agent, category, role] for agent selection."""
        if self.phase == "full" and self._trust_state:
            return trust_state_to_scores(self._trust_state)
        return self._scores or {}

    def update(
        self,
        agent_answers: Dict[str, str],
        final_answer: str,
        gt_answer: str,
        category: str,
        agent_roles: Dict[str, str],
    ) -> Dict[str, float]:
        """
        한 턴 업데이트. Returns scaled_rewards R̃_i.
        """
        # Phase 1: rewards
        rewards = compute_rewards(
            agent_answers, final_answer, gt_answer, kappa=self.kappa
        )

        # N_c 증가
        self.N_c[category] = self.N_c.get(category, 0) + 1
        N_c = self.N_c[category]

        # Phase 2: scaling
        scaled = scale_rewards(rewards, N_c, T=self.T)

        if self.phase == "simple":
            self._scores = update_scores_simple(
                self._scores,
                scaled,
                category,
                agent_roles,
                gamma=self.gamma,
            )
        else:
            self._trust_state = update_credibility_full(
                self._trust_state,
                scaled,
                category,
                agent_roles,
                lambda_f=self.lambda_f,
                lambda_g=self.lambda_g,
                mu=self.mu,
                gamma=self.gamma,
            )

        return scaled
