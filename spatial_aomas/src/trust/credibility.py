"""
Phase 3: Credibility Update (Beta + EMA).

- n+_{i,k,c}(t+1) = n+ + r̃,  n-_{i,k,c}(t+1) = n- + (1-r̃)
  r̃ ∈ [0,1]: R̃를 [0,1]로 매핑. r̃ = (R̃+1)/2
- q = n+ / (n+ + n-)
- f = (1-λ_f)*f + λ_f*R̃
- g = (1-λ_g)*g + λ_g*q
- s̃ = μ*f + (1-μ)*g
- s = s̃ + γ*R̃

Ref: https://arxiv.org/pdf/2403.16956.pdf (Beta trust)
Ref: RLJ 2024 (fast/slow EMA)
"""
import copy
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrustState:
    """
    Per (agent, role, category) state.
    """
    n_plus: float = 0.5
    n_minus: float = 0.5
    f: float = 0.5  # 단기 EMA
    g: float = 0.5  # 장기 EMA
    s: float = 0.5  # 최종 score


def _reward_to_01(R_tilde: float) -> float:
    """R̃ ∈ [-1,1] → r̃ ∈ [0,1] for Beta update."""
    return max(0.0, min(1.0, (R_tilde + 1.0) / 2.0))


def update_credibility_full(
    state: Dict[str, Dict[str, Dict[str, TrustState]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    lambda_f: float = 0.3,
    lambda_g: float = 0.1,
    mu: float = 0.5,
    gamma: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, TrustState]]]:
    """
    Full credibility update for each (agent, role, category).

    Args:
        state: {agent: {category: {role: TrustState}}}
        scaled_rewards: {agent_id: R̃_i}
        category: c
        agent_roles: {agent_id: role}
    """
    out = copy.deepcopy(state)

    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, "Direct")
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = TrustState()

        t = out[agent_id][category][role]
        r_tilde = _reward_to_01(R_tilde)

        # Beta update
        t.n_plus = t.n_plus + r_tilde
        t.n_minus = t.n_minus + (1.0 - r_tilde)

        # q = n+ / (n+ + n-)
        denom = t.n_plus + t.n_minus
        q = t.n_plus / denom if denom > 0 else 0.5

        # f, g EMA
        t.f = (1.0 - lambda_f) * t.f + lambda_f * R_tilde
        t.g = (1.0 - lambda_g) * t.g + lambda_g * q

        # s̃ = μ*f + (1-μ)*g
        s_tilde = mu * t.f + (1.0 - mu) * t.g
        t.s = max(0.0, min(1.0, s_tilde + gamma * R_tilde))

    return out


def trust_state_to_scores(
    state: Dict[str, Dict[str, Dict[str, TrustState]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract s scores from TrustState for agent selection."""
    return {
        agent: {
            cat: {role: t.s for role, t in roles.items()}
            for cat, roles in cats.items()
        }
        for agent, cats in state.items()
    }
