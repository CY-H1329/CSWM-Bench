# Spatial_AOMAS — Trust Score Module

Trust Score 업데이트 로직 (3단계 점진적 구현).

## Main Architecture에서 import

**단일 파일 `trust_score.py`** 에 모든 함수 포함:

```python
from spatial_aomas.trust_score import (
    step1_compute_rewards,   # Phase 1: GT vs R_i
    step2_scale_rewards,     # Phase 2: R̃_i = φ·R_i
    step3_update_scores_simple,   # Phase 3: s += γ·R̃
    step4_update_credibility_full,  # Phase 4: Beta + EMA
    get_scores_from_state,   # TrustState → s 테이블
    trust_update,            # 통합: 한 번에 호출
    TrustState,
)
```

## 구조

```
spatial_aomas/
├── trust_score.py    # ★ 단일 파일 (Main Architecture에서 import)
├── config.py
├── example_usage.py
├── README.md
└── src/trust/        # (기존 분리 모듈, 호환용)
```

## 사용법

```python
# 개별 단계 호출
rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=1.0)
scaled = step2_scale_rewards(rewards, N_c, T=10.0)
state = step4_update_credibility_full(state, scaled, category, agent_roles)
scores = get_scores_from_state(state)

# 또는 통합 호출
updated, scaled = trust_update(
    agent_answers, final_answer, gt_answer, category, agent_roles,
    scores_or_state=state, N_c=N_c, phase="full",
)
scores = get_scores_from_state(updated)
```
