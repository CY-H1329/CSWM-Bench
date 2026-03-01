"""
Phase 2: Category 샘플 수 기반 Reward Scaling.

- N_c: category c에서 누적 샘플 수
- φ(N_c) = 1 - exp(-N_c/T)
- R̃_i = φ(N_c) · R_i
- s_{i,k,c}^{(t+1)} = s_{i,k,c}^{(t)} + γ·R̃_i
"""
import copy
import math
from typing import Dict


def phi_scale(N_c: int, T: float) -> float:
    """
    φ(N_c) = 1 - exp(-N_c/T)
    초기에는 작은 업데이트, 후반에는 강한 분리.
    """
    if T <= 0:
        return 1.0
    return 1.0 - math.exp(-N_c / T)


def scale_rewards(
    rewards: Dict[str, float],
    N_c: int,
    T: float = 10.0,
) -> Dict[str, float]:
    """
    R̃_i = φ(N_c) · R_i
    """
    phi = phi_scale(N_c, T)
    return {agent_id: phi * R_i for agent_id, R_i in rewards.items()}


def update_scores_simple(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    gamma: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    s_{i,k,c}^{(t+1)} = s_{i,k,c}^{(t)} + γ·R̃_i^{(t)}

    agent_roles: {agent_id: role} — 각 agent가 맡은 역할
    """
    out = copy.deepcopy(scores)

    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, "Direct")
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = 0.5
        s_old = out[agent_id][category][role]
        s_new = max(0.0, min(1.0, s_old + gamma * R_tilde))
        out[agent_id][category][role] = s_new

    return out
