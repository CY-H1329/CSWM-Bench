"""
Spatial AOMAS — Trust Score 모듈.

메인 아키텍처에서 score 기반 agent 선택 및 Step 1~4 업데이트.
"""
from .trust_score import (
    ROLES,
    select_agents_by_score,
    step1_compute_rewards,
    step2_scale_rewards,
    step2_phi_scale,
    step3_update_scores_simple,
    step4_update_credibility_full,
    get_scores_from_state,
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    similarity_answer,
    TrustState,
)

__all__ = [
    "ROLES",
    "select_agents_by_score",
    "step1_compute_rewards",
    "step2_scale_rewards",
    "step2_phi_scale",
    "step3_update_scores_simple",
    "step4_update_credibility_full",
    "get_scores_from_state",
    "run_step1",
    "run_step2",
    "run_step3",
    "run_step4",
    "similarity_answer",
    "TrustState",
]
