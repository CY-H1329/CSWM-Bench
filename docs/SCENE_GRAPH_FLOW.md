# scene_graph_construction 플로우: 관계 기반 추론

이미지 + 쿼리가 들어왔을 때 **scene_graph_construction** ROLE이 관계를 강력하게 추출하고, 그 관계 그래프로 추론하는 전체 플로우입니다.

---

## 1. 목표

| 목표 | 설명 |
|------|------|
| **관계 강력 추출** | 이미지 + 쿼리에서 질문 관련 객체를 포함해 **구조화된 관계 그래프** 생성 |
| **그래프 기반 추론** | 노드/엣지 구조를 활용해 **traversal 절차**로 답 도출 |
| **Cypher 생략** | 객체 수가 적으므로 JSON + 프롬프트 기반 추론으로 충분 |

---

## 2. 전체 플로우 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: image (PIL), query (str)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: 관계 강력 추출 (Tool)                                              │
│  ─────────────────────────────────────                                      │
│  1) 쿼리/이미지에서 객체 추출 (Open-vocab)                                     │
│  2) 객체 검출 (OWL-ViT 또는 DETR) → bbox                                      │
│  3) bbox 기하로 pairwise 관계 계산 (above, below, left_of, right_of, overlaps)│
│  4) 구조화된 JSON (nodes + edges) 출력                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: 그래프 기반 추론 (Agent)                                            │
│  ─────────────────────────────────────                                      │
│  1) 질문에서 기준 객체·관계 식별                                               │
│  2) 그래프 traversal: 해당 relation 엣지 따라가기                              │
│  3) 도달한 노드/집합 → 답                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Answer + Reason → SharedMemory                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: 관계 강력 추출 (상세)

### 3.1 트리거 (Pipeline)

`assignments`에 `(scene_graph_construction, llm_name)`이 포함되면 실행됩니다.

```python
# pipeline.py (개선 시)
if role == "scene_graph_construction":
    object_names = extract_objects_from_image(image, specialist_generate, llm_name)
    tool_output_cache[role] = get_scene_graph(image, object_names=object_names)
```

---

### 3.2 Step 1: 객체 추출 (VLM, 쿼리 인지)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  extract_objects_from_image(image, specialist_generate, llm_name)            │
│  (explicit_3d와 동일한 함수 사용)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VLM + 프롬프트                                                               │
│  "List all distinct objects visible in this image. Include objects that      │
│   might be relevant to spatial questions (above/below, left/right, etc).     │
│   Output comma-separated."                                                    │
│  → image + prompt → "chair, table, person, window, bottle, ..."              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    object_names = ["chair", "table", "person", "window", ...]
```

**핵심:** COCO 80 고정이 아니라 **Open-vocab** — 질문에 나오는 객체도 포함.

---

### 3.3 Step 2: 검출 + 관계 계산 (Tool)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  get_scene_graph(image, object_names)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Sub-Tool 1: get_detections_  │         │  Sub-Tool 2: Pairwise         │
│  with_labels (OWL-ViT)        │         │  Spatial Relations           │
│                               │         │                               │
│  • object_names → bboxes      │         │  • bbox center (cx, cy)       │
│  • Fallback: DETR             │         │  • cy_a < cy_b → above       │
│  • objects: [{id, label, box}]│         │  • cx_a < cx_b → left_of      │
└──────────────┬───────────────┘         │  • IoU > 0.05 → overlaps      │
               │                         └──────────────┬─────────────────┘
               │                                        │
               └────────────────┬───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  구조화된 JSON 생성 (nodes + edges)                                            │
│                                                                              │
│  nodes = [{"id": "1", "label": "chair", "bbox": [...], "center": [cx,cy]}, ...]│
│  edges = [{"subject": "1", "relation": "above", "object": "2"}, ...]          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 Step 3: Tool 출력 형식 (구조화된 JSON)

```
## Scene Graph Tool Output (structured)

### Nodes (objects)
| id | label  | bbox        | center   |
|----|--------|-------------|----------|
| 1  | chair  | [x1,y1,x2,y2] | [cx,cy] |
| 2  | table  | [...]       | [...]    |
| 3  | person | [...]       | [...]    |

### Edges (pairwise spatial relations)
| subject | relation | object |
|---------|----------|--------|
| 1       | above    | 2      |
| 1       | left_of  | 2      |
| 3       | left_of  | 1      |
| 2       | below    | 3      |

### Traversal Protocol
- "A is above B" → find edges where subject=A, relation=above, object=B
- "What is left of X?" → find edges where subject=?, relation=left_of, object=X
- "What is above X?" → find edges where subject=?, relation=above, object=X
```

---

## 4. Phase 2: 그래프 기반 추론 (상세)

### 4.1 Prompt 구성 (build_role_prompt)

```python
# prompts.py - build_role_prompt(role="scene_graph_construction", query, tool_output)
tool_section = "## Tool Output (structured graph)\n\n" + tool_output
role_prompt = template.format(query=query, tool_section=tool_section)
```

**프롬프트 핵심:**
- Tool 출력 = **nodes + edges** 구조화된 형태
- **Traversal Protocol** 명시: "left of X" → left_of 엣지에서 object=X인 것의 subject 찾기

---

### 4.2 Agent 추론 절차 (VLM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  질문: "Which object is to the left of the chair?"                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 질문 파싱                                                            │
│  • 기준 객체: chair                                                           │
│  • 관계: left of                                                             │
│  • 질문 형태: "What is [relation] [object]?" → subject 찾기                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 그래프 Traversal                                                     │
│  • relation = left_of, object = chair (id=1)                                 │
│  • edges에서 (subject=?, relation=left_of, object=1) 검색                     │
│  • → subject = 3 (person)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 노드 → 답                                                            │
│  • node 3 = person                                                           │
│  • 답: person (옵션에 맞게 (B) 등으로 매핑)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 데이터 흐름 다이어그램

```
                    ┌──────────────┐
                    │  image       │
                    │  query       │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │ extract_objects_     │   │ query (prompt에)    │
    │ from_image(image)    │   │                     │
    │ → object_names       │   │                     │
    └──────────┬──────────┘   └──────────┬──────────┘
               │                         │
               ▼                         │
    ┌─────────────────────┐              │
    │ get_scene_graph     │              │
    │ (image, object_names)               │
    │                     │              │
    │ • OWL-ViT/DETR      │              │
    │ • pairwise relations│              │
    │ • nodes + edges JSON│              │
    └──────────┬──────────┘              │
               │                         │
               ▼                         ▼
    ┌─────────────────────────────────────────────┐
    │  build_role_prompt(role, query, tool_output) │
    │  = ROLE + structured graph + query + format  │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  specialist_generate(llm, image, role_prompt)│
    │  → Traversal protocol 따라 추론              │
    │  → Answer: (X) + Reason: ...                │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  SharedMemory.add(scene_graph, answer, reason)│
    └─────────────────────────────────────────────┘
```

---

## 6. explicit_3d vs scene_graph 비교

| 항목 | explicit_3d_representation | scene_graph_construction |
|------|---------------------------|---------------------------|
| **추출** | VLM 객체 → OWL-ViT → Depth | VLM 객체 → OWL-ViT → Relations |
| **데이터** | z (depth), ordering, count | nodes, edges (above, left_of 등) |
| **추론** | z 비교, Instance Count | 그래프 traversal |
| **강점** | closer/farther, in front/behind | above/below, left/right |
| **좌표계** | 3D (depth) | 2D (이미지 평면) |

---

## 7. 구현 체크리스트

| 단계 | 현재 | 개선 |
|------|------|------|
| 객체 추출 | DETR COCO 80만 | extract_objects_from_image (VLM) + OWL-ViT |
| 관계 계산 | bbox 기하 (유지) | bbox 기하 (유지) |
| 출력 형식 | 플랫 텍스트 | **구조화된 JSON** (nodes + edges) |
| 프롬프트 | "Traverse the graph" (그래프 없음) | **Traversal Protocol** 명시 |
| Pipeline | get_scene_graph_summary(image) | get_scene_graph(image, object_names) |

---

## 8. 예시: "Which object is above the table?"

**입력:**
- 이미지: 의자, 테이블, 사람, 창문
- 쿼리: "Which object is above the table? (A) chair (B) person (C) window (D) none"

**Phase 1 출력 (Tool):**
```json
{
  "nodes": [
    {"id": "1", "label": "chair", "center": [120, 380]},
    {"id": "2", "label": "table", "center": [200, 420]},
    {"id": "3", "label": "person", "center": [180, 280]},
    {"id": "4", "label": "window", "center": [200, 80]}
  ],
  "edges": [
    {"subject": "3", "relation": "above", "object": "2"},
    {"subject": "4", "relation": "above", "object": "3"},
    {"subject": "1", "relation": "left_of", "object": "2"}
  ]
}
```

**Phase 2 추론:**
1. 질문: "above the table" → relation=above, object=table(id=2)
2. edges에서 (subject=?, relation=above, object=2) → subject=3
3. node 3 = person
4. 답: (B) person

---

## 9. 관련 파일 (개선 후)

| 파일 | 역할 |
|------|------|
| `src2/tools/object_extraction.py` | extract_objects_from_image (VLM) |
| `src2/tools/scene_graph.py` | get_scene_graph(image, object_names) — 구조화된 JSON |
| `src2/tools/open_vocab_detection.py` | OWL-ViT (object_names로 검출) |
| `src2/agents/mas_v2/pipeline.py` | scene_graph 툴 호출 시 object_names 전달 |
| `src2/agents/mas_v2/prompts.py` | scene_graph ROLE + Traversal Protocol |
