#!/usr/bin/env python3
"""
Spatial_AOMAS Trust Score — 참고용 예시.

실행은 메인 아키텍처에서 이루어짐.
이 파일은 함수 사용법 확인용.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatial_aomas.trust_score import (
    ROLES,
    select_agents_by_score,
    step1_compute_rewards,
    step2_scale_rewards,
    step3_update_scores_simple,
    step4_update_credibility_full,
    get_scores_from_state,
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    TrustState,
)

AGENTS = ["qwen3_4b", "sa2va", "spatialreasoner"]
CATEGORIES = ["depth", "relation", "count"]


def main():
    agent_answers = {
        "qwen3_4b": "Final Answer: (A)",
        "sa2va": "The answer is (B)",
        "spatialreasoner": "Answer: (A)",
    }
    final_answer = "Final Answer: (A)"
    gt_answer = "(A)"
    category = "depth"
    agent_roles = {
        "qwen3_4b": "Direct",
        "sa2va": "3D",
        "spatialreasoner": "SceneGraph",
    }

    # 초기 점수 (예시)
    scores = {
        a: {c: {r: 0.5 for r in ROLES} for c in CATEGORIES}
        for a in AGENTS
    }

    # 3 roles에 3 agents 배정
    role_to_agent = select_agents_by_score(scores, category, AGENTS)
    print("select_agents_by_score:", role_to_agent)

    # Step1: 보상 그대로 점수에 추가
    scores = run_step1(scores, agent_answers, final_answer, gt_answer, category, agent_roles)
    print("run_step1 scores:", scores)

    # Step2: 스케일된 보상 점수에 추가
    scores = run_step2(scores, agent_answers, final_answer, gt_answer, category, agent_roles, N_c=1)
    print("run_step2 scores:", scores)

    # Step3: gamma로 조절해서 점수 누적
    scores = run_step3(scores, agent_answers, final_answer, gt_answer, category, agent_roles, N_c=1)
    print("run_step3 scores:", scores)

    # Step4 — scores 테이블을 마지막에 in-place 갱신
    state = {a: {c: {r: TrustState() for r in ROLES} for c in CATEGORIES} for a in AGENTS}
    scores = {a: {c: {r: 0.5 for r in ROLES} for c in CATEGORIES} for a in AGENTS}
    state = run_step4(state, scores, agent_answers, final_answer, gt_answer, category, agent_roles, N_c=1)
    print("run_step4 scores:", scores)


if __name__ == "__main__":
    main()
