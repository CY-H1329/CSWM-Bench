"""
Spatial_AOMAS confidenc Score — 메인 아키텍처용 단일 파일.

역할 3가지:
  - Direct: Direct Visual Heuristic Strategy
  - 3D: Explicit 3D Representation Construction
  - SceneGraph: Scene Graph Construction

핵심 함수:
  - select_agents_by_score(): score 기반으로 3 roles에 agents 배정
  - step1_compute_rewards(): 각 agent가 맞췄는지 보상 계산
  - step2_scale_rewards(): 샘플 수에 따라 보상 스케일 조절
  - step3_update_scores_simple(): 단순 점수 누적 업데이트
  - step4_update_credibility_full(): Beta+EMA 기반 신뢰도 업데이트
"""
import copy
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 역할 정의 (3가지)
# ---------------------------------------------------------------------------

ROLES = [
    "Direct",        # Direct Visual Heuristic Strategy
    "3D",            # Explicit 3D Representation Construction
    "SceneGraph",    # Scene Graph Construction
]

# MAS v2 pipeline role names (5 unified categories와 함께 사용)
ROLES_MASV2 = [
    "direct_visual_heuristic",
    "explicit_3d_representation",
    "scene_graph_construction",
]


# ---------------------------------------------------------------------------
# 메인 아키텍처: Score 기반 Agent 선택
# ---------------------------------------------------------------------------

def select_agents_by_score(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    category: str,
    candidate_agents: List[str],
    roles: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    각 역할(role)마다 가장 점수가 높은 agent를 1명씩 배정한다.
    결과: 3 roles → 3 agents (역할당 1:1)

    scores[agent][category][role] = 점수
    반환: {role: agent} — 예) {"Direct": "qwen3", "3D": "sa2va", ...}
    roles: None이면 ROLES 사용, MAS v2 연동 시 ROLES_MASV2 전달.
    """
    roles = list(roles or ROLES)
    agents = list(candidate_agents)
    # agent 수가 role 수보다 적으면 role을 줄임
    if len(agents) < len(roles):
        roles = roles[: len(agents)]
    if not roles:
        return {}

    assignment: Dict[str, str] = {}
    used_agents: set = set()

    for role in roles:
        best_agent = None
        best_score = -1e9
        for agent in agents:
            if agent in used_agents:
                continue  # 이미 다른 role에 배정된 agent는 제외
            s = scores.get(agent, {}).get(category, {}).get(role, 0.5)
            if s > best_score:
                best_score = s
                best_agent = agent
        if best_agent is not None:
            assignment[role] = best_agent
            used_agents.add(best_agent)  # 중복 배정 방지
        elif agents:
            assignment[role] = agents[0]
    return assignment


# ---------------------------------------------------------------------------
# 답변 비교 유틸
# ---------------------------------------------------------------------------

def _normalize_for_comparison(s: str) -> str:
    """문자열을 비교하기 쉽게: 공백 제거, 소문자, 연속 공백 하나로."""
    if not s:
        return ""
    return " ".join((s or "").strip().lower().split())


def _extract_answer(text: str) -> str:
    """
    텍스트에서 답변 부분만 뽑는다.
    (A), Answer: X, 마지막 토큰 등 패턴 시도.
    """
    text = (text or "").strip()
    if not text:
        return ""
    tail = text[-500:] if len(text) > 500 else text
    # 괄호 안 내용: (A), (42) 등
    m = re.search(r"\(([^)]+)\)", tail)
    if m:
        return m.group(1).strip()
    # Answer: 뒤 내용
    m = re.search(r"(?:final\s*answer|answer)[:\s]+([^\n,]+)", tail, re.I)
    if m:
        return m.group(1).strip()
    # 마지막 알파벳/숫자 토큰
    tokens = re.findall(r"[A-Za-z0-9]+", tail)
    return tokens[-1] if tokens else ""


def similarity_answer(pred: str, gt: str, normalize_fn: Optional[Callable[[str], str]] = None) -> float:
    """
    예측 답과 정답(GT)이 같은지 본다.
    같으면 1.0, 다르면 0.0, GT 없으면 0.5 (애매함).
    """
    if normalize_fn is not None:
        p, g = normalize_fn(pred), normalize_fn(gt)
    else:
        gt_part = _extract_answer(gt) or gt
        g = _normalize_for_comparison(gt_part)
        if not g:
            return 0.5
        pred_part = _extract_answer(pred) or pred
        p = _normalize_for_comparison(pred_part)
    if not g:
        return 0.5
    return 1.0 if p == g else 0.0


# ---------------------------------------------------------------------------
# Step 1: 보상 계산
# ---------------------------------------------------------------------------

def step1_compute_rewards(
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    kappa: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, float]:
    """
    각 agent가 GT와 얼마나 맞췄는지 점수화한다.
    맞으면 +1에 가깝고, 틀리면 -1에 가깝다.
    최종 답이 틀렸을 때, 맞은 agent보다 틀린 agent에게 더 큰 패널티.
    """
    sim_fn = similarity_fn or similarity_answer
    final_correct = sim_fn(final_answer, gt_answer) >= 0.99
    sim_final = sim_fn(final_answer, gt_answer)

    rewards = {}
    for agent_id, answer in agent_answers.items():
        sim_i = sim_fn(answer, gt_answer)
        R_i = 2.0 * sim_i - 1.0  # sim=1이면 +1, sim=0이면 -1
        if not final_correct:  # 최종 답 틀림 → 틀린 agent에게 추가 패널티
            delta = max(0.0, sim_final - sim_i)
            R_i = R_i - kappa * delta
        rewards[agent_id] = R_i
    return rewards


# ---------------------------------------------------------------------------
# Step 2: 보상 스케일 조절
# ---------------------------------------------------------------------------

def step2_phi_scale(N_c: int, T: float) -> float:
    """
    이 카테고리에서 본 샘플 수(N_c)가 많을수록 보상 반영을 강하게 한다.
    초반에는 0에 가깝고, 후반에 1에 수렴.
    """
    return 1.0 if T <= 0 else 1.0 - math.exp(-N_c / T)


def step2_scale_rewards(
    rewards: Dict[str, float],
    N_c: int,
    T: float = 10.0,
) -> Dict[str, float]:
    """
    Step1 보상을 스케일한다.
    샘플이 적을 때는 업데이트를 부드럽게, 많을 때는 강하게.
    """
    phi = step2_phi_scale(N_c, T)  # 0~1, 샘플 많을수록 1에 가까움
    return {agent_id: phi * R_i for agent_id, R_i in rewards.items()}


# ---------------------------------------------------------------------------
# Step 3: 단순 점수 업데이트
# ---------------------------------------------------------------------------

def step3_update_scores_simple(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    gamma: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    각 agent의 점수를 보상만큼 더한다.
    새 점수 = 기존 점수 + gamma * 스케일된 보상
    """
    out = copy.deepcopy(scores)
    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, ROLES[0])
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = 0.5  # 초기값
        s_old = out[agent_id][category][role]
        out[agent_id][category][role] = s_old + gamma * R_tilde  # 점수 누적
    return out


# ---------------------------------------------------------------------------
# Step 4: Beta + EMA 신뢰도 업데이트
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceState:
    """
    한 (agent, role, category)의 신뢰 상태.
    n_plus/n_minus: 맞은/틀린 횟수 누적
    f: 최근 성능, g: 장기 성능
    s: 최종 점수 (agent 선택에 사용)
    """
    n_plus: float = 0.5
    n_minus: float = 0.5
    f: float = 0.5
    g: float = 0.5
    s: float = 0.5


def _reward_to_01(R_tilde: float) -> float:
    """보상 [-1,1]을 [0,1]로 바꾼다 (Beta 분포용)."""
    return max(0.0, min(1.0, (R_tilde + 1.0) / 2.0))


def step4_update_credibility_full(
    state: Dict[str, Dict[str, Dict[str, ConfidenceState]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    lambda_f: float = 0.3,
    lambda_g: float = 0.1,
    mu: float = 0.5,
    gamma: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, ConfidenceState]]]:
    """
    Beta 분포와 EMA로 신뢰도를 업데이트한다.
    단기(f)와 장기(g)를 섞어서 최종 점수(s)를 만든다.
    """
    out = copy.deepcopy(state)
    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, ROLES[0])
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = ConfidenceState()

        t = out[agent_id][category][role]
        r_tilde = _reward_to_01(R_tilde)  # [-1,1] → [0,1]

        t.n_plus += r_tilde
        t.n_minus += (1.0 - r_tilde)
        q = t.n_plus / (t.n_plus + t.n_minus) if (t.n_plus + t.n_minus) > 0 else 0.5  # 맞은 비율

        t.f = (1.0 - lambda_f) * t.f + lambda_f * R_tilde  # 단기: 최근 보상
        t.g = (1.0 - lambda_g) * t.g + lambda_g * q       # 장기: 누적 비율
        s_tilde = mu * t.f + (1.0 - mu) * t.g             # 단기·장기 혼합
        t.s = s_tilde + gamma * R_tilde
    return out


def get_scores_from_state(
    state: Dict[str, Dict[str, Dict[str, ConfidenceState]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    ConfidenceState에서 점수(s)만 뽑아서 select_agents_by_score에 넘길 수 있게 한다.
    """
    return {
        agent: {cat: {role: t.s for role, t in roles.items()} for cat, roles in cats.items()}
        for agent, cats in state.items()
    }


# ---------------------------------------------------------------------------
# Step별 편의 함수 (메인 아키텍처에서 그대로 호출)
# ---------------------------------------------------------------------------

def run_step1(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    category: str,
    agent_roles: Dict[str, str],
    kappa: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Step1: 보상 계산 후 점수에 그대로 더해서 업데이트. (s += R)"""
    rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=kappa)
    return step3_update_scores_simple(scores, rewards, category, agent_roles, gamma=1.0)


def run_step2(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    category: str,
    agent_roles: Dict[str, str],
    N_c: int,
    kappa: float = 1.0,
    T: float = 10.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Step2: 보상 스케일 후 점수에 그대로 더해서 업데이트. (s += R̃)"""
    rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=kappa)
    scaled = step2_scale_rewards(rewards, N_c, T=T)
    return step3_update_scores_simple(scores, scaled, category, agent_roles, gamma=1.0)


def run_step3(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    category: str,
    agent_roles: Dict[str, str],
    N_c: int,
    kappa: float = 1.0,
    T: float = 10.0,
    gamma: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Step1+2+3: 보상 계산 → 스케일 → 점수 업데이트. 단순 누적."""
    rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=kappa)
    scaled = step2_scale_rewards(rewards, N_c, T=T)
    return step3_update_scores_simple(scores, scaled, category, agent_roles, gamma=gamma)


def run_step4(
    state: Dict[str, Dict[str, Dict[str, ConfidenceState]]],
    scores: Dict[str, Dict[str, Dict[str, float]]],
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    category: str,
    agent_roles: Dict[str, str],
    N_c: int,
    kappa: float = 1.0,
    T: float = 10.0,
    gamma: float = 0.1,
    lambda_f: float = 0.3,
    lambda_g: float = 0.1,
    mu: float = 0.5,
) -> Dict[str, Dict[str, Dict[str, ConfidenceState]]]:
    """
    Step1+2+3+4: 보상 → 스케일 → Beta+EMA 신뢰도 업데이트.
    마지막에 scores 테이블을 갱신한다 (in-place).
    Returns: updated_state
    """
    rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=kappa)
    scaled = step2_scale_rewards(rewards, N_c, T=T)
    updated_state = step4_update_credibility_full(
        state, scaled, category, agent_roles,
        lambda_f=lambda_f, lambda_g=lambda_g, mu=mu, gamma=gamma,
    )
    # scores 테이블 갱신 (in-place)
    for agent, cats in updated_state.items():
        if agent not in scores:
            scores[agent] = {}
        for cat, roles in cats.items():
            if cat not in scores[agent]:
                scores[agent][cat] = {}
            for role, t in roles.items():
                scores[agent][cat][role] = t.s
    return updated_state
