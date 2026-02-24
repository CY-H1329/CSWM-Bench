# explicit_3d_representation 에이전트: Fallback 설계 & 태스크 강점

## 1. 도구 실패 시 Fallback 설계

### 현재 동작

| 상황 | tool_output | 에이전트가 받는 것 |
|------|-------------|-------------------|
| **정상** | Depth Map Grid + z + Count + ... | 전체 3D representation |
| **Depth 실패** | `[3D tool: Depth estimation failed. Proceed with visual analysis only.]` | 메시지 + 이미지로 추론 |
| **객체 0개** | `[3D tool: No objects detected. Proceed with visual analysis only.]` | 메시지 + 이미지로 추론 |
| **Pipeline 예외** | `""` (빈 문자열) | tool_section 없음 |

### Fallback 강화 (Prompt)

도구가 없을 때는 **Pictorial Cues**로 추론:

- **Occlusion**: A가 B를 가림 → A가 더 가까움
- **Relative size**: 더 크게 보이면 더 가까움
- **Height in image**: 화면 아래에 더 가까움 (일반적)
- **Count**: 영역별 스캔, 단위 정의, semantic match

→ **항상 답변**하도록 설계. 도구 없을 때는 "Tool unavailable; reasoning from image"라고 명시.

---

## 2. 어떤 태스크에 강한가?

### 3D 표현 (도구 정상 시)

| 태스크 | 강도 | 이유 |
|--------|------|------|
| **Depth** | ★★★★★ | z 값, Depth Map Grid로 직접 비교 |
| **Distance** | ★★★★★ | "which is closer/farther" → z 비교 |
| **In front/behind** | ★★★★★ | Depth Ordering, Δz로 명확 |
| **Count** | ★★★★☆ | Instance Count + VLM 객체 추출. OWL-ViT 검출 품질에 의존 |
| **Relation (above/below)** | ★★★☆☆ | 2D 위치 위주. Grid의 region은 일부 보조 |

### 도구 실패 시 (Pictorial Cues)

| 태스크 | 강도 | 이유 |
|--------|------|------|
| **Depth** | ★★★☆☆ | Occlusion, size, height로 추론 가능 |
| **Distance** | ★★★☆☆ | 위와 동일 |
| **In front/behind** | ★★★☆☆ | Occlusion이 핵심 |
| **Count** | ★★★☆☆ | 스캔 + 단위 정의 |
| **Relation (above/below)** | ★★★★☆ | 2D 이미지 위치가 직접적 |

### 3DSRBench / CV-Bench

| 벤치마크 | explicit_3d 강점 | 비고 |
|----------|------------------|------|
| **3DSRBench** | location_closer_to_camera, multi_object_closer_to, orientation_in_front_of | depth/거리 중심 |
| **CV-Bench** | Depth, Distance | 3D 질문에 적합 |
| **CV-Bench** | Count | Instance Count + open-vocab detection |
| **CV-Bench** | Relation | 2D 관계는 scene_graph가 더 적합 |

---

## 3. 요약

- **도구 정상**: Depth, Distance, In front/behind → 매우 강함 (z, Δz, Grid 사용)
- **도구 실패**: Pictorial Cues로 fallback, 항상 답변 시도
- **Count**: 도구 정상 시 Instance Count로 강함; 실패 시 스캔 기반으로 보완
- **2D Relation**: 위/아래/좌우는 scene_graph가 더 적합
