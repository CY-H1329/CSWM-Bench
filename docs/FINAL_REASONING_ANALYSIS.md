# Final Reasoning Agent 성능 분석

## 현상

- **direct_visual_heuristic** (단일 specialist): CV-Bench ~84.5% (1000 샘플)
- **MAS v2 Final Reasoning** (3 specialists → synthesis): CV-Bench ~85.4% (96 샘플)

→ Final Reasoning이 direct_visual보다 크게 나아지지 않음. 오히려 direct_visual이 더 좋아 보이는 경우도 있음.

---

## 원인 분석

### 1. direct_visual baseline이 이미 강함

| 카테고리 | direct_visual (CV-Bench) |
|----------|--------------------------|
| Count    | 70.7%                    |
| Depth    | 91.9%                    |
| Distance | 88.1%                    |
| Relation | 88.7%                    |
| **Overall** | **84.5%**            |

- CV-Bench는 **pictorial cues**(occlusion, size, height in image)로 풀 수 있는 문제가 많음
- direct_visual은 이미지를 직접 보고 추론 → 정보 손실 없음
- 개선 여지가 작음 (ceiling effect)

---

### 2. Final Reasoning Agent는 이미지를 보지 못함 (text-only)

```
Specialists (VLM)     →  이미지 + 쿼리  →  답 + 이유 (텍스트)
Final Reasoning       →  텍스트만 (SharedMemory)  →  synthesis
```

- Final Reasoning은 **이미지를 전혀 보지 못함**
- specialist들의 텍스트 설명만 받음
- specialist가 잘못 추론했을 때, Final Reasoning이 이를 검증할 방법이 없음
- "어느 specialist가 맞는지"를 **텍스트만으로** 판단해야 함 → 어려움

---

### 3. 다른 specialist들이 노이즈를 추가

| Specialist | 강점 | CV-Bench에서의 실제 |
|------------|------|---------------------|
| direct_visual | Count, Depth, Relation | ✅ pictorial cues로 대부분 커버 |
| explicit_3d | Depth, Distance | ⚠️ depth tool 실패/노이즈, z값 부정확 |
| scene_graph | Relation (above/below, left/right) | ⚠️ OWL-ViT 오검출, 엣지 오류 |

- **explicit_3d**: `get_3d_representation`이 실패하거나 depth 추정이 부정확하면 잘못된 답
- **scene_graph**: `get_scene_graph`의 pairwise relation이 틀리면 잘못된 답
- direct_visual이 맞는데, explicit_3d/scene_graph가 틀리면 → Final Reasoning이 **틀린 쪽을 선택**할 수 있음
- "Do not blindly follow majority"라고 해도, **텍스트만으로** 어느 쪽이 맞는지 구분하기 어려움

---

### 4. 질문 유형 vs specialist 전략 매칭 부족

- Final Reasoning 프롬프트에 "which agent's strategy matches the question"을 강조하지만,
- **실제로** 질문 유형(Count/Depth/Relation)과 specialist 전략이 잘 매칭되어도,
- specialist 출력이 틀리면 synthesis가 틀릴 수밖에 없음
- 반대로, direct_visual이 이미지를 보고 한 번에 맞추면 → 중간 synthesis 단계가 오히려 **혼란**을 줄 수 있음

---

### 5. 모델 역량

- **DeepSeek-R1-Distill-Qwen-7B**: full DeepSeek-R1의 distilled 버전
- 복잡한 synthesis (3개 입력 비교, 최선 선택)가 direct inference보다 어려울 수 있음
- 7B distilled가 "어느 specialist가 이 질문에 더 적합한가"를 정확히 판단하기 어려울 수 있음

---

### 6. Agent 선택 로직 (Score Map)

- 현재: **항상 3 specialist 모두** 실행
- train phase 없이 test만 하면 score map이 초기값(0.5) 유지
- 어떤 질문에 어떤 specialist가 유리한지 **학습되지 않음**
- 일부 질문에서는 explicit_3d, scene_graph가 오히려 노이즈만 추가

---

## 요약: 왜 Final Reasoning이 크게 오르지 않는가?

| 요인 | 설명 |
|------|------|
| **Baseline 강함** | direct_visual이 이미 84.5%로 높음 |
| **정보 비대칭** | Final Reasoning은 이미지 없음, specialist 텍스트만 의존 |
| **노이즈** | explicit_3d, scene_graph가 틀리면 synthesis가 틀린 쪽을 선택 |
| **Synthesis 난이도** | "3개 중 최선" 선택이 direct inference보다 어려움 |
| **모델** | Distill-7B가 복잡한 synthesis에 한계 |

---

## 개선 방향 (제안)

### A. 질문 유형별 specialist 가중치

- Count → direct_visual 신뢰도 ↑
- Depth/Distance → explicit_3d 신뢰도 ↑ (tool 성공 시)
- Relation (above/below, left/right) → scene_graph 신뢰도 ↑

→ Final Reasoning 프롬프트에 "For Count questions, direct_visual is usually most reliable" 등 **전략별 신뢰도 힌트** 추가

### B. Tool 실패 시 명시

- SharedMemory에 `[Tool failed]` 등 표시
- Final Reasoning이 "이 agent는 tool 없이 추론했음"을 알 수 있게 함
- tool 실패 시 해당 specialist 신뢰도 낮추도록 유도

### C. Specialist 1개만 사용 (ablation)

- direct_visual만 사용했을 때 vs 3 specialist 사용 시 비교
- 3 specialist가 오히려 성능을 떨어뜨리는지 확인

### D. Final Reasoning에 이미지 제공 (구조 변경)

- VLM 기반 Final Reasoning으로 변경 (이미지 + SharedMemory)
- 비용/지연 증가하지만, 검증 능력 향상

### E. Train phase 활용

- score map을 train으로 학습시켜, 카테고리별로 유리한 specialist 선택
- test 시 학습된 score map 사용

---

## 참고: nomal.md 결과

```
direct_visual_heuristic (1000 samples): 84.5%
MAS v2 Final Reasoning (96 samples):     85.4%
```

→ 샘플 수 차이 있음. 동일 샘플로 비교 시 차이가 더 명확해질 수 있음.
