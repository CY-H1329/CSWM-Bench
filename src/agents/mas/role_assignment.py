"""
Role assignment: score-based (s[agent, role, category]).

Role is NOT chosen by the agent — assigned by ScoreManager based on s[i,k,c].
i_k* = argmax_i s[i,k,c] for each role k.
"""
from typing import Dict, List, Optional

from .config import MAS_ROLES
from .score_manager import ScoreManager


def assign_roles_from_scores(
    score_manager: ScoreManager,
    category: str,
    candidate_agents: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Assign each role to the best agent by score.
    Returns {role: agent} for role in MAS_ROLES.
    """
    return score_manager.assign_roles(category, candidate_agents)


def agent_to_role_mapping(role_to_agent: Dict[str, str]) -> Dict[str, str]:
    """
    Invert {role: agent} → {agent: role}.
    Used to know which role each selected agent plays.
    """
    return {agent: role for role, agent in role_to_agent.items()}


def get_scores_for_display(
    score_manager: ScoreManager,
    category: str,
    agents: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Get {agent: {role: score}} for pipeline / Head prompt.
    """
    return score_manager.get_role_scores_table(category)
