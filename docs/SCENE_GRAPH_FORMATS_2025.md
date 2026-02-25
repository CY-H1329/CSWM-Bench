# 2025–2026 논문들의 관계 그래프 형식

## 1. 최소 공통 구조 (De Facto Standard)

대부분의 연구가 공유하는 **최소한의** 관계 그래프 구조:

```
노드(Node) = 객체 (id, label, [위치/속성])
엣지(Edge) = 관계 (subject_id, relation_type, object_id)
```

---

## 2. 2D Scene Graph (이미지 기반)

### 2.1 GQA / Visual Genome 스타일 (표준)

**형식**: JSON, triplet `(subject, predicate, object)`

```json
{
  "objects": {
    "271881": {
      "name": "chair",
      "x": 220, "y": 310,
      "w": 50, "h": 80,
      "attributes": ["brown", "wooden", "small"],
      "relations": {
        "32452": {"name": "on", "object": "275312"},
        "32453": {"name": "left of", "object": "279472"}
      }
    }
  }
}
```

**관계 예시**: `on`, `left of`, `right of`, `above`, `below`, `holding`, `near`, `in`, `wearing` 등

---

### 2.2 Universal Scene Graph (USG, CVPR 2025)

- **노드**: 여러 modality(텍스트, 이미지, 비디오, 3D)에서 온 객체를 **통합 노드**로 병합
- **속성**: 각 modality의 segmentation mask 유지
- **엣지**: intra-modality + inter-modality 관계

---

## 3. 3D Scene Graph (공간/로봇)

### 3.1 계층적 3D Scene Graph (MIT, arXiv 2025)

**형식**: Graph DB (Neo4j) + **Cypher** 쿼리 인터페이스

**레이어 구조** (Objects → Places → Regions):

| 레이어 | 노드 | 엣지 |
|--------|------|------|
| **Objects** | semantic label, position, bbox | `on top of`, relational |
| **Mesh Places** | semantic label, position, boundary | traversable connectivity |
| **3D Places** | position, bounding sphere | connectivity |
| **Room/Region** | semantic label, position | containment |

**노드 예시**:
- Object: `{id, class, position, bbox}`
- Place: `{id, sibling_places, parent_region}`
- Region: `{id, label, position}`

**인터페이스**: LLM이 **Cypher**로 그래프 쿼리 → DB가 결과 반환 → 추론

---

### 3.2 Functional 3D Scene Graph (OpenFunGraph, CVPR 2025)

**추가 요소**:
- **Interactive elements**: 스위치, 손잡이 등
- **Functional relations**: `switch controls light`, `door opens to room`

**형식**: 기존 spatial relation + functional relation

---

## 4. 2025–2026 트렌드 요약

| 항목 | 2025–2026 최소 기대 |
|------|---------------------|
| **구조** | 노드(객체) + 엣지(관계), **구조화된 표현** (JSON/Graph DB) |
| **노드** | id, label, position/bbox, [attributes] |
| **엣지** | (subject, relation_type, object) triplet |
| **관계 종류** | spatial (above, left_of, on) + semantic (holding, wearing) + [functional] |
| **인터페이스** | 직렬화 텍스트 → **Graph DB + 쿼리 언어**(Cypher) |
| **객체** | Open-vocab (질문 객체 포함) |

---

## 5. 우리 구현에 적용할 최소 형식

### 5.1 2D Spatial Graph (이미지 VQA용)

```json
{
  "nodes": [
    {"id": "1", "label": "chair", "bbox": [x1,y1,x2,y2], "center": [cx,cy]},
    {"id": "2", "label": "table", "bbox": [...], "center": [...]}
  ],
  "edges": [
    {"subject": "1", "relation": "above", "object": "2"},
    {"subject": "1", "relation": "left_of", "object": "2"}
  ]
}
```

**관계**: `above`, `below`, `left_of`, `right_of`, `overlaps` (2D spatial)

### 5.2 추론 인터페이스 옵션

| 옵션 | 복잡도 | 설명 |
|------|--------|------|
| **A. JSON 텍스트** | 낮음 | 구조화된 JSON을 프롬프트에 넣고 LLM이 "traversal" 절차 따르도록 유도 |
| **B. Cypher** | 높음 | Graph DB + Cypher tool → LLM이 쿼리 생성 |
| **C. Python API** | 중간 | `graph.get_neighbors(node, relation="left_of")` 등 함수 제공 |

---

## 6. 참고 문헌

- **GQA** (CVPR 2019): JSON scene graph format
- **Visual Genome**: (subject, predicate, object) triplet
- **Structured Interfaces for 3D Scene Graphs** (arXiv 2025): Cypher + Graph DB
- **Universal Scene Graph** (CVPR 2025): Multimodal unified nodes
- **OpenFunGraph** (CVPR 2025): Functional 3D scene graphs
