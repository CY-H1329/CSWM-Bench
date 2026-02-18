# Head-Agent 5가지 핵심 능력 평가

Head-Agent 선정을 위한 5가지 능력 테스트.

## 1. Task Decomposition (문제 분류)
- **역할**: 문제를 정확히 분류
- **위험**: 잘못 분류 시 trust 학습이 왜곡됨
- **평가**: GT category와 일치 여부 (accuracy)
- **기존**: `run_eval_category_routing.py`와 동일

## 2. Routing Decision (에이전트 선택)
- **역할**: 어떤 agent를 고를지 (Direct / Perception / Reasoning / Both)
- **의미**: specialization 학습의 시작점
- **평가**: 유효한 4가지 옵션 중 하나 출력 여부 (format validity)

## 3. Complexity Estimation (복잡도 판단)
- **역할**: 간단(1) vs 복잡(5) 판단
- **의미**: tool 과용 방지, shortcut 방지
- **평가**: 1-5 스케일 유효 출력 여부 (format validity)

## 4. Strategy Planning (전략 초안)
- **역할**: tool/strategy 초안 제시
- **의미**: Perception policy의 출발점
- **평가**: 번호 매긴 단계 + 관련 키워드 포함 여부

## 5. Trust-Aware Logging (구조화된 trace)
- **역할**: reasoning trace를 구조화
- **의미**: 추후 trust update 신호 확보
- **평가**: 유효 JSON + 필수 키(reasoning, category, route, confidence 등) 포함 여부

---

## 실행

```bash
# 5가지 능력 모두 (cvbench, 50 samples)
python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --max_samples 50

# 특정 능력만
python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --capability routing --max_samples 30
python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --capability trust_logging --model claude_opus_4_5

# cvbench + 3dsrbench
python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --benchmark all --max_samples 50
```

## 출력 구조

```
results/runs/head_agent/
├── cvbench/
│   ├── task_decomposition/<timestamp>/
│   ├── routing/<timestamp>/
│   ├── complexity/<timestamp>/
│   ├── strategy/<timestamp>/
│   └── trust_logging/<timestamp>/
└── 3dsrbench/
    └── ...
```

각 capability별로 `summary.txt`, `{model}/details.jsonl`, `{model}/results.json` 저장.
