# Spatial AOMAS — Trust Score Module

Trust Score 업데이트 로직 (Step 1~4). 메인 아키텍처에서 score 기반 agent 선택 및 업데이트.

## 역할 4가지

- **Direct**: Direct Visual Heuristic Strategy
- **3D**: Explicit 3D Representation Construction
- **SceneGraph**: Scene Graph Construction
- **MentalTransform**: Mental Transformation

## 사용법

```python
from spatial_aomas import (
    ROLES,
    select_agents_by_score,
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    get_scores_from_state,
    TrustState,
)

# 4 roles에 4 agents 배정
role_to_agent = select_agents_by_score(scores, category, candidate_agents)

# Step별 점수 업데이트
scores = run_step1(scores, agent_answers, final_answer, gt_answer, category, agent_roles)
scores = run_step2(scores, agent_answers, final_answer, gt_answer, category, agent_roles, N_c)
scores = run_step3(scores, agent_answers, final_answer, gt_answer, category, agent_roles, N_c, gamma=0.1)

# Step4 (Beta+EMA)
state = {a: {c: {r: TrustState() for r in ROLES} for c in CATEGORIES} for a in AGENTS}
state = run_step4(state, agent_answers, final_answer, gt_answer, category, agent_roles, N_c)
scores = get_scores_from_state(state)
```

## Step 요약

| Step | 설명 |
|------|------|
| 1 | 보상 R 계산 후 `s += R` |
| 2 | 스케일된 보상 R̃ 계산 후 `s += R̃` |
| 3 | `s += γ·R̃` (gamma로 조절) |
| 4 | Beta + EMA 신뢰도 업데이트 |

## 예시 실행

```bash
cd Spatial_MAS
python spatial_aomas/example_usage.py
```
