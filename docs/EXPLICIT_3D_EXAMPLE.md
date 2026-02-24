# explicit_3d_representation 에이전트 예시

이미지 + 쿼리가 주어졌을 때, **explicit_3d** 에이전트가 어떤 데이터를 받고 어떻게 추론하는지 단계별 예시입니다.

---

## 입력 (벤치마크에서)

**이미지:** 거실 장면 (소파, 커피 테이블, 창문, 쓰레기통 1개, 사람 1명)

**쿼리:**
```
How many trash cans are in the image?
Options:
(A) 0
(B) 1
(C) 2
(D) 3
```

---

## Step 1: 객체 추출 (VLM + 프롬프트)

**VLM 입력:**
- Image: [거실 장면 이미지]
- Prompt: "List all distinct objects visible in this image. Include furniture, people, vehicles, animals, containers, and any other notable items. Output ONLY a comma-separated list of object names..."

**VLM 출력 (예시):**
```
couch, coffee table, person, trash can, window, plant, lamp
```

**파싱 결과:**
```python
object_names = ["couch", "coffee table", "person", "trash can", "window", "plant", "lamp"]
```

---

## Step 2: OWL-ViT 검출

**입력:** image + candidate_labels = ["couch", "coffee table", "person", "trash can", "window", "plant", "lamp"]

**출력 (bbox + label + score):**
```
couch:      (50, 200, 400, 450), score=0.92
coffee table: (180, 350, 320, 420), score=0.88
person:     (250, 150, 380, 420), score=0.91
trash can:  (420, 380, 480, 480), score=0.75
window:     (100, 0, 350, 180), score=0.85
plant:      (30, 300, 120, 450), score=0.72
```

---

## Step 3: Depth + bbox → 3D Representation Tool 출력

**Depth map:** DepthAnything으로 각 bbox 영역의 mean depth 계산

**정렬 (closer → farther):**
- person (bottom-center) — mean_depth 낮음
- coffee table (center)
- couch (mid-left)
- trash can (bottom-right)
- plant (mid-left)
- window (top-center) — mean_depth 높음

**Tool 최종 출력 (텍스트):**

```
## 3D Representation Tool Output (mathematical depth)

### 1. Depth Map Grid (3×3, normalized [0=closest, 1=farthest])
  top-left:0.82  top-center:0.75  top-right:0.88
  mid-left:0.45  center:0.32     mid-right:0.51
  bottom-left:0.18  bottom-center:0.12  bottom-right:0.25

### 2. Object Depth Values (z ∈ [0, 1], 0=closest)
  1. person (bottom-center): z=0.000 [CLOSEST]
  2. coffee table (center): z=0.185
  3. couch (mid-left): z=0.312
  4. trash can (bottom-right): z=0.428
  5. plant (mid-left): z=0.556
  6. window (top-center): z=1.000 [FARTHEST]

### 3. Depth Ordering (adjacent pairs)
  - person (z=0.00) → coffee table (z=0.19)  [Δz=0.19]
  - coffee table (z=0.19) → couch (z=0.31)  [Δz=0.12]
  - couch (z=0.31) → trash can (z=0.43)  [Δz=0.12]
  - trash can (z=0.43) → plant (z=0.56)  [Δz=0.13]
  - plant (z=0.56) → window (z=1.00)  [Δz=0.44]

### 4. Instance Count by Object Type
  - person: 1
  - coffee table: 1
  - couch: 1
  - trash can: 1
  - plant: 1
  - window: 1

### Mathematical Interpretation
  - z: normalized depth (0=closest, 1=farthest). Monocular depth, relative scale.
  - Δz: depth difference. Larger Δz = greater separation in depth.
  - Depth Map Grid: spatial layout. Map object position to grid region.
```

---

## Step 4: 에이전트가 받는 전체 Prompt

```
# ROLE: Explicit 3D Representation Construction Agent

You are a **3D depth reasoning specialist**. You answer spatial questions (closer/farther, in front/behind, depth order) using the **object-level 3D representation** provided below...

## Tool Output (use this data in your reasoning)

[위 Step 3의 Tool 출력 전체]

## How to Use the 3D Representation Tool Output
...

## Task
Question: How many trash cans are in the image?
Options:
(A) 0
(B) 1
(C) 2
(D) 3

## Output Format (STRICT)
...
```

---

## Step 5: 에이전트 추론 과정 (VLM 내부)

**질문:** "How many trash cans?"

**Protocol 적용:**
1. **Identify question objects** → trash can
2. **Match to tool output** → Instance Count에 "trash can: 1" 존재
3. **For "how many X?"** → Instance Count 사용, trash can = 1

**추론:**
- Tool의 Instance Count: trash can: 1
- 질문 객체 "trash cans" ≈ "trash can" (semantic match)
- Count = 1 → 옵션 (B)

---

## Step 6: 에이전트 출력

```
Answer: (B)

Reason:
[1] Question objects: trash cans
[2] From tool Instance Count: trash can: 1
[3] Semantic match: "trash cans" ≈ "trash can"
[4] Conclusion → 1 trash can → option (B)
```

---

## 다른 질문 예시: Depth

**쿼리:**
```
Which object is closer to the camera?
Options:
(A) The person
(B) The window
(C) The trash can
(D) The plant
```

**추론:**
1. Question objects: person, window, trash can, plant
2. Object Depth Ordering: person(1) < plant(5) < trash can(4) < window(6)
3. Rank 1 = person = CLOSEST
4. Conclusion → (A)

**출력:**
```
Answer: (A)

Reason:
[1] Question objects: person, window, trash can, plant
[2] From tool: person rank 1 (CLOSEST), window rank 6 (FARTHEST)
[3] Depth inference: person is closest to camera
[4] Conclusion → option (A)
```

---

## 또 다른 예시: "In front of"

**쿼리:**
```
Is the trash can in front of the person?
Options:
(A) Yes
(B) No
```

**추론:**
1. Object Depth Values: person z=0.00, trash can z=0.43
2. z_person < z_trash_can → person이 trash can보다 앞(가까움)
3. → trash can은 person 뒤에 있음
4. "trash can in front of person" = No

**출력:**
```
Answer: (B)

Reason:
[1] Question: Is trash can in front of person?
[2] From tool: person z=0.00, trash can z=0.43. z_person < z_trash_can.
[3] So trash can is BEHIND person, not in front
[4] Conclusion → No → option (B)
```

---

## 요약

| 단계 | 입력 | 출력 |
|------|------|------|
| 1. 객체 추출 | image + VLM prompt | ["couch", "coffee table", "person", "trash can", ...] |
| 2. OWL-ViT | image + object_names | bboxes for each |
| 3. 3D Tool | depth + bboxes | Object Ordering + Pairwise + Instance Count |
| 4. Agent | image + tool_output + query | Answer + Reason |

**핵심:** 고정 목록 없이 VLM이 이미지에서 객체를 뽑고, OWL-ViT가 검출하고, depth와 결합해 3D representation을 만든 뒤, 에이전트가 이 정보로 Count/Depth/Relation을 추론합니다.
