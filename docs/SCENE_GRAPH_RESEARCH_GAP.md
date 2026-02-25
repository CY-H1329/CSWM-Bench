# Scene Graph: 연구 기대 vs 현재 구현

## 1. 연구에서 Scene Graph가 필요한 이유

### 1.1 핵심 논점

이미지는 **객체들의 나열**이 아니라 **객체 간 관계의 네트워크**다.  
연구에서는 이를 **구조화된 그래프 표현**으로 다루고, **그래프 기반 추론**으로 질문에 답한다.

### 1.2 연구 기대 흐름

```
이미지 → Scene Graph (노드=객체, 엣지=관계)
         ↓
쿼리 → 그래프 상에서의 추론 연산 (traversal, message passing)
         ↓
답변
```

- **GraphVQA** (ACL 2021): 질문을 **GNN message passing 연산**으로 번역·실행  
  - 예: "What is the red object left of the girl holding a hamburger?"  
  - → message passing: hamburger → girl → red tray  
  - → 최종 노드 상태 = 답 (tray)  
  - GQA 88.43% → **94.78%** (SOTA 대비 큰 향상)

- **Graph-Structured Representations** (CVPR 2017):  
  - scene 객체 + 질문 단어를 **그래프**로 표현  
  - CNN/LSTM 대신 **그래프 구조**를 활용해 추론

- **VQA-GNN**:  
  - scene graph + concept graph를 **multimodal message passing**으로 융합  
  - VCR +3.2%, GQA +4.6%

### 1.3 연구에서의 역할

| 단계 | 연구 기대 | 설명 |
|------|-----------|------|
| 1. **구조화** | Scene Graph = 노드 + 엣지 | 객체=노드, 관계=엣지, 그래프 자료구조 |
| 2. **쿼리 매핑** | 질문 → 그래프 연산 | "left of X" → left_of 엣지 따라가기 |
| 3. **그래프 추론** | GNN / traversal / message passing | 구조를 이용한 명시적 추론 |
| 4. **답변** | 최종 노드/집합 → 답 | 그래프 연산 결과로 답 생성 |

---

## 2. 현재 구현의 한계

### 2.1 현재 흐름

```
이미지 → DETR → 텍스트 요약 (객체 목록 + pairwise 관계 나열)
         ↓
VLM: 텍스트 + 이미지 + 쿼리 → 답변
```

- **구조화**: 그래프 자료구조 없음, **플랫 텍스트**만 존재
- **쿼리 매핑**: "left of X" → left_of 엣지 따라가기 없음
- **그래프 추론**: GNN/message passing 없음, VLM이 **텍스트를 읽고 추론**

### 2.2 구체적 한계

| 항목 | 연구 기대 | 현재 구현 |
|------|-----------|-----------|
| **출력 형태** | 그래프 (adjacency, 노드/엣지) | 텍스트 요약 |
| **추론 방식** | 그래프 연산 (traversal, GNN) | VLM 텍스트 이해 |
| **쿼리 활용** | 질문 → 그래프 연산으로 실행 | 질문을 프롬프트에 넣고 VLM에 위임 |
| **객체 범위** | Open-vocab (질문 객체 포함) | COCO 80 클래스 (DETR) |
| **관계 종류** | semantic (holding, wearing 등) | 기하만 (above, left_of, overlaps) |

### 2.3 프롬프트 vs 실제

프롬프트에는 다음이 적혀 있다:

> "Traverse the graph to answer the question"

하지만:

- **실제 그래프가 없음** → traversal 불가
- **텍스트만** 있음 → VLM이 자연어로 추론
- 결과적으로 **그래프 기반 추론이 아님**

---

## 3. 연구 기대에 맞추려면

### 3.1 구조화된 그래프 제공

- 노드: `{id, label, (cx, cy)}`
- 엣지: `(A, relation, B)` 예: `(chair, above, table)`
- **형식**: JSON, adjacency list, 또는 Cypher 등 쿼리 가능한 형태

### 3.2 쿼리 → 그래프 연산 매핑

- "left of X" → `left_of` 엣지로 X에서 이웃 탐색
- "above Y" → `above` 엣지로 Y에서 이웃 탐색
- "between A and B" → A–B 경로/중간 노드 탐색

### 3.3 그래프 기반 추론

**옵션 A: GNN (연구 스타일)**  
- 질문 → instruction vectors  
- GNN message passing으로 그래프 상 추론  
- 최종 노드/집합 → 답

**옵션 B: LLM + 구조화된 그래프 (현실적)**  
- 그래프를 JSON/텍스트로 제공  
- 프롬프트에 "traversal 절차" 명시  
  - 예: "1) 질문에서 기준 객체 찾기 2) 해당 관계 엣지 따라가기 3) 도달한 객체가 답"  
- LLM이 이 절차를 따라 추론하도록 유도

**옵션 C: 그래프 쿼리 언어**  
- Cypher 등으로 그래프 쿼리  
- LLM이 쿼리 생성 → 실행 → 결과 해석

### 3.4 Open-vocab 객체

- explicit_3d처럼 **VLM + OWL-ViT**로 질문 관련 객체 추출
- DETR COCO 80만으로는 "hamburger", "tray" 등 누락 가능

---

## 4. 요약

| 질문 | 답 |
|------|-----|
| **연구에서 scene graph가 필요한 이유?** | 이미지를 **관계 그래프**로 표현하고, **그래프 연산**으로 추론하기 위해 |
| **연구 기대 역할?** | 관계 그래프 구축 → 이미지+쿼리에 대해 **그래프 기반 추론** → 답 |
| **현재 구현의 한계?** | 그래프 구조 없음, 텍스트만 제공, VLM이 텍스트로 추론 (그래프 연산 아님) |
| **개선 방향?** | 1) 구조화된 그래프 출력 2) 쿼리→그래프 연산 매핑 3) traversal/GNN 또는 LLM 기반 그래프 추론 절차 4) Open-vocab 객체 |

---

## 5. 참고 문헌

- **GraphVQA** (ACL 2021): Language-guided GNN, message passing으로 질문 실행
- **Graph-Structured Representations for VQA** (CVPR 2017): 객체+질문 그래프, 구조 활용
- **VQA-GNN**: Scene graph + concept graph, multimodal message passing
- **Structured Interfaces for 3D Scene Graphs** (2025): Cypher 등 쿼리 언어로 그래프 접근
