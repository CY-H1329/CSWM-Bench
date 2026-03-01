"""
Trust Score: reward computation, scaling, credibility update.
"""
from .reward import compute_rewards, similarity_answer
from .scaling import scale_rewards, phi_scale, update_scores_simple
from .credibility import update_credibility_full, TrustState, trust_state_to_scores
from .manager import TrustManager

__all__ = [
    "compute_rewards",
    "similarity_answer",
    "scale_rewards",
    "phi_scale",
    "update_scores_simple",
    "update_credibility_full",
    "TrustState",
    "trust_state_to_scores",
    "TrustManager",
]
