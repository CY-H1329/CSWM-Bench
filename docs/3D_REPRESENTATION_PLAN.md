# 3D Representation Agent 완벽 구축 플랜

## 목표

explicit_3d_representation 에이전트가 **객체 단위 depth**를 활용해 3DSRBench, CV-Bench 등 다양한 벤치마크에서 정확한 spatial reasoning을 수행하도록 구축.

---

## 현재 한계

| 항목 | 현재 | 한계 |
|------|------|------|
| Depth tool | 3×3 region (9개 영역) | 객체별 depth 없음. "chair vs table which closer?" 풀기 어려움 |
| 객체 정보 | scene_graph에만 있음 | depth와 분리됨 |
| Agent 추론 | region → 객체 매핑을 agent가 추측 | 부정확, 오류 누적 |

---

## 핵심 아이디어: Object-level 3D Representation

**DETR (객체 검출) + DepthAnything (depth map) → 각 객체 bbox 내 mean depth → 객체별 depth 순서**

```
이미지 → [DETR] → 객체들 (chair, table, ...) + bbox
       → [DepthAnything] → depth map (H×W)
       → 각 bbox 영역의 depth mean → 객체별 depth 값
       → 정렬 (closer → farther) → "chair(1) < table(2) < window(3)"
```

---

## 구현 단계

### Phase 1: Tool 확장

1. **depth.py**: `get_depth_map(image) -> np.ndarray` 추가
   - 기존 pipeline 재사용, depth 배열만 반환
   - get_depth_summary는 내부적으로 이걸 사용

2. **scene_graph.py**: `get_detected_objects(image) -> List[dict]` 추가
   - DETR 결과: `{id, label, box, score}` 리스트
   - get_scene_graph_summary는 이걸 사용

3. **3d_representation.py** (신규)
   - `get_3d_representation(image) -> str`
   - depth_map + objects → 객체별 depth 계산
   - 출력: 객체명, 위치, depth 순위, pairwise "A in front of B"

### Phase 2: Tool 출력 포맷

```
## 3D Representation Tool Output

### Object-level Depth Ordering (closer → farther from camera)
1. chair (bottom-center) — CLOSEST
2. dining table (center)
3. person (mid-left)
4. window (top-center) — FARTHEST

### Pairwise Depth Relations (for question resolution)
- chair is IN FRONT OF dining table
- dining table is IN FRONT OF window
- chair is IN FRONT OF window

### Region Summary (3×3 backup)
Closer regions: bottom-center, center
Farther regions: top-center, top-right
```

### Phase 3: Pipeline 연동

- `explicit_3d_representation` role → `get_3d_representation(image)` 호출 (기존 `get_depth_summary` 대체)
- 또는 두 tool 모두 제공하고 agent prompt에서 선택? → **get_3d_representation만 사용** (더 풍부한 정보)

### Phase 4: Agent Prompt 강화

- 객체 단위 depth가 있으므로 "Map objects to regions" 단계 불필요
- 직접 "chair(1) < table(2) → chair is closer" 사용
- Pairwise relations 활용: "Is A in front of B?" → tool에서 직접 확인

---

## 벤치마크 커버리지

| 질문 유형 | 3D Tool 활용 | 예시 |
|-----------|--------------|------|
| Depth | ✅ 직접 | "Which is closer to camera?" → 객체 순서에서 선택 |
| Distance | ✅ 직접 | "Which is farther?" → 순서 역순 |
| In front/behind | ✅ 직접 | "Is A in front of B?" → pairwise relation |
| Relation (above/below) | ⚠️ 2D | depth + scene_graph의 above/below 조합 |
| Count | ❌ | scene_graph 또는 direct_visual이 담당 |
| Orientation | ❌ | direct_visual, scene_graph |

---

## 모델

- Specialist: **Qwen3-VL-4B** (기존과 동일)
- Tool: DepthAnything-Small, DETR-ResNet50 (기존과 동일, 재사용)

---

## 예상 효과

1. **Depth/Distance/In-front 질문**: 객체 단위 depth로 직접 해결 → 정확도 향상
2. **Agent 추론 부담 감소**: region 매핑 추측 불필요
3. **Final Reasoning Agent**: explicit_3d의 답이 더 구체적 → 합의 시 신뢰도 상승
