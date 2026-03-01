"""
Spatial_AOMAS — Trust Score module.
"""
from .trust.reward import compute_rewards, similarity_answer
from .trust.scaling import scale_rewards, update_scores_simple, phi_scale
from .trust.credibility import update_credibility_full, TrustState, trust_state_to_scores

__all__ = [
    "compute_rewards",
    "similarity_answer",
    "scale_rewards",
    "phi_scale",
    "update_scores_simple",
    "update_credibility_full",
    "TrustState",
    "trust_state_to_scores",
]
