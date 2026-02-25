# scene_graph_construction 에이전트 역할

## 1. 설계상 역할

**scene_graph_construction**은 **2D 이미지 평면에서의 공간 관계**를 다루는 specialist입니다.

- **객체(노드)** + **쌍별 공간 관계(엣지)** 로 scene graph 구성
- **above/below, left/right, overlaps** 등 **2D bbox 기반** 관계 제공
- **3D depth는 다루지 않음** (explicit_3d 담당)

---

## 2. 3명 Specialist 역할 분담

| Specialist | 담당 | 데이터 소스 |
|------------|------|-------------|
| **direct_visual_heuristic** | Pictorial cues, Count, 일반 | 이미지만 |
| **explicit_3d_representation** | Depth, Distance, In front/behind, Count | Depth map + z + Instance Count |
| **scene_graph_construction** | **2D 관계**: above/below, left/right, overlaps | Bbox 기반 기하 |

---

## 3. scene_graph가 강한 Spatial Reasoning

### Unified category 기준

| Category | scene_graph 강점 | 이유 |
|----------|------------------|------|
| **spatial_relation** | ★★★★★ | above/below, next to → bbox의 cy, cx 비교 |
| **orientation** (left/right) | ★★★★☆ | left_of, right_of 직접 제공 |

### Fine-grained category (3DSRBench)

| Category | 설명 | scene_graph |
|----------|------|-------------|
| **location_above** | A가 B 위/아래 | above/below from bbox |
| **location_next_to** | 인접 | bbox 거리로 추론 |
| **orientation_on_the_left** | 왼쪽/오른쪽 | left_of/right_of |

### CV-Bench

| Task | 설명 | scene_graph |
|------|------|-------------|
| **Relation** | left/right, above/below, inside/outside | 2D 관계에 직접 대응 |

---

## 4. Tool 출력 (get_scene_graph_summary)

```
Detected Objects: id, label, position (upper-left, center, ...)
Pairwise Spatial Relationships:
  chair(1) — above, left_of — table(2)
  table(2) — below — window(3)
  ...
```

**관계 종류:** 2D bbox에서 추출

- **above** / **below**: cy (y좌표) 비교
- **left_of** / **right_of**: cx (x좌표) 비교
- **overlaps** (possible occlusion): IoU > 0.05

---

## 5. explicit_3d와의 차이

| 항목 | explicit_3d | scene_graph |
|------|-------------|-------------|
| **좌표계** | 3D (depth z) | 2D (이미지 평면) |
| **관계** | in front/behind, closer/farther | above/below, left/right, overlaps |
| **질문 예** | "Which is closer?" | "Is A above B?" |
| **Tool** | Depth + OWL-ViT (open-vocab) | DETR (COCO 80) |

---

## 6. Agent 선택

ScoreMap은 **모든 category에 대해 3명 specialist를 모두 실행**합니다.

- category별로 role별 최적 LLM만 선택
- **scene_graph는 항상 실행**됨
- Final Reasoning Agent가 3명의 답을 합쳐 최종 답 선택

---

## 7. 요약

**scene_graph_construction**은:

- **2D spatial relation** (above/below, left/right, overlaps) 전문
- **Relation**, **location_above**, **orientation_on_the_left** 등에 적합
- Depth와는 분리 (explicit_3d가 3D 정보 담당)
