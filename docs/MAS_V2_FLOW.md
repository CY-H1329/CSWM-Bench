# MAS v2 플로우: 이미지 + 쿼리 → 최종 답변

벤치마크에서 **이미지 + 쿼리**가 들어왔을 때 전체 파이프라인이 어떻게 동작하는지 정리합니다.

---

## 1. 진입점: 벤치마크 → 데이터 로드

```
run_eval_mas_v2.py
    └── load_benchmark(benchmark="3dsrbench" | "cvbench")
            └── src2/benchmarks/loaders.py
                    ├── HuggingFace dataset 로드 (ccvl/3DSRBench, nyu-visionx/CV-Bench)
                    └── example = { image, question, choices, answer, category, ... }
```

**데이터 추출 함수:**
- `get_benchmark_image(example, benchmark)` → PIL Image
- `get_benchmark_prompt(example, benchmark)` → "질문\nOptions:\n(A) ...\n(B) ..."
- `get_benchmark_answer(example, benchmark)` → "A" | "B" | "C" | "D"

---

## 2. 샘플별 실행: run_step(image, query, gt, ...)

각 샘플마다 `run_step`이 한 번 호출됩니다.

```
example = dataset[step]
image   = get_benchmark_image(example, benchmark)   # PIL Image
query   = get_benchmark_prompt(example, benchmark)   # 질문 + 옵션
gt      = get_benchmark_answer(example, benchmark)  # 정답 (A/B/C/D)

run_step(image=image, query=query, gt=gt, ...)
```

---

## 3. run_step 내부 플로우

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: image (PIL), query (str), gt (str)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Head Agent (Qwen3-VL-4B)                                            │
│  ─────────────────────────────────                                          │
│  Input:  image + head_prompt (query만 포함, 5개 카테고리 설명)                  │
│  Output: category (spatial_relation | distance_depth | size | orientation |   │
│                   counting)                                                  │
│                                                                              │
│  head_prompt = build_head_agent_prompt(query, ALL_CATEGORIES, ...)            │
│  head_raw    = head_generate(image, head_prompt)                              │
│  category    = parse_category(head_raw, ALL_CATEGORIES)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Agent Selection (ScoreMap)                                          │
│  ─────────────────────────────────                                          │
│  assignments = score_map.select_agents(category, step)                       │
│  → [(role, llm), (role, llm), (role, llm)]  (3쌍, role당 1개)                 │
│                                                                              │
│  예: [(direct_visual_heuristic, qwen3_4b),                                   │
│       (explicit_3d_representation, qwen3_4b),                                │
│       (scene_graph_construction, sa2va)]                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Specialist Agents (3명, 병렬 개념이지만 순차 실행)                     │
│  ─────────────────────────────────────────────────────────                  │
│                                                                              │
│  for (role, llm_name) in assignments:                                        │
│                                                                              │
│    [Tool 사용 여부]                                                           │
│    - explicit_3d_representation → get_3d_representation(image)               │
│    - scene_graph_construction   → get_scene_graph_summary(image)             │
│    - direct_visual_heuristic     → tool 없음                                  │
│                                                                              │
│    role_prompt = build_role_prompt(role, query, tool_output=tool_output)      │
│    raw_output  = specialist_generate(llm_name, image, role_prompt)          │
│    answer, reason = parse_specialist_output(raw_output)                       │
│    shared_memory.add(role, llm_name, answer, reason)                         │
│                                                                              │
│  각 specialist 입력:                                                         │
│    - image: 동일한 벤치마크 이미지                                             │
│    - role_prompt: query + (해당 role용 tool 출력, 있으면)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Final Reasoning Agent (DeepSeek-R1, text-only)                      │
│  ─────────────────────────────────────────────────────                     │
│  Input:  query + SharedMemory 텍스트                                          │
│          (3명 specialist의 Answer + Reasoning)                              │
│  Output: final_answer (A/B/C/D)                                               │
│                                                                              │
│  reasoning_prompt = build_final_reasoning_prompt(query, shared_memory_text)   │
│  reasoning_raw    = reasoning_generate(reasoning_prompt)  ← 이미지 없음!      │
│  final_answer     = parse_final_answer(reasoning_raw)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Score Map Update (Train 시에만)                                      │
│  ────────────────────────────────────                                      │
│  updater.update(score_map, category, assignments, agent_results,             │
│                 final_answer, gt, step, total_steps)                          │
│  → 각 (role, llm) 조합의 점수 갱신                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: { final_answer, correct, agent_details, ... }                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 이미지/쿼리 사용처 요약

| 단계 | 이미지 | 쿼리 | 비고 |
|------|--------|------|------|
| Head Agent | ✅ | ✅ | image + query → category |
| direct_visual_heuristic | ✅ | ✅ (prompt에) | tool 없음 |
| explicit_3d_representation | ✅ (tool에도) | ✅ (prompt에) | get_3d_representation(image) |
| scene_graph_construction | ✅ (tool에도) | ✅ (prompt에) | get_scene_graph_summary(image) |
| Final Reasoning | ❌ | ✅ | text-only, SharedMemory만 사용 |

---

## 5. 데이터 흐름 다이어그램

```
                    ┌──────────────┐
                    │  Benchmark   │
                    │  (example)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  image  │  │  query  │  │   gt    │
         └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              │            │            │ (eval only)
              ▼            ▼            │
    ┌─────────────────────────────┐    │
    │      Head Agent (VLM)       │    │
    │   image + query → category  │    │
    └──────────────┬──────────────┘    │
                   │                   │
                   ▼                   │
    ┌──────────────────────────────┐   │
    │  ScoreMap.select_agents()    │   │
    │  → 3 (role, llm) pairs       │   │
    └──────────────┬───────────────┘   │
                   │                   │
     ┌─────────────┼─────────────┐     │
     ▼             ▼             ▼     │
┌─────────┐  ┌──────────┐  ┌──────────┐│
│direct   │  │explicit  │  │scene     ││
│visual   │  │3d        │  │graph     ││
│         │  │          │  │          ││
│image +  │  │image +   │  │image +   ││
│prompt   │  │3d_tool + │  │sg_tool + ││
│         │  │prompt    │  │prompt    ││
└────┬────┘  └────┬─────┘  └────┬─────┘│
     │            │              │     │
     └────────────┼──────────────┘     │
                  ▼                    │
         ┌────────────────┐           │
         │  SharedMemory   │           │
         │  (3 answers +   │           │
         │   reasons)      │           │
         └────────┬────────┘           │
                  │                    │
                  ▼                    │
         ┌────────────────┐           │
         │ Final Reasoning│           │
         │ query + memory │           │
         │ → final_answer │           │
         └────────┬───────┘           │
                  │                    │
                  ▼                    ▼
         ┌────────────────────────────────┐
         │  correct = (final_answer==gt)   │
         └─────────────────────────────────┘
```

---

## 6. explicit_3d_representation 에이전트 상세 플로우

이미지 + 쿼리가 들어왔을 때 **explicit_3d** 에이전트가 어떻게 처리하는지 단계별로 정리합니다.

---

### 6.1 트리거 조건

`assignments`에 `(explicit_3d_representation, llm_name)`이 포함되면 이 에이전트가 실행됩니다.  
(ScoreMap이 category별로 3개 role을 선택하므로, 대부분의 샘플에서 실행됨)

---

### 6.2 Step 1: Tool 호출 (Pipeline에서)

```python
# pipeline.py
if role == "explicit_3d_representation":
    object_names = extract_objects_from_image(image, specialist_generate, llm_name)
    tool_output_cache[role] = get_3d_representation(image, object_names=object_names)
```

**흐름:** 1) VLM으로 이미지에서 객체 목록 추출 → 2) 그 목록으로 3D representation 생성

---

### 6.3 Step 2a: 객체 추출 (VLM + 프롬프트)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  extract_objects_from_image(image, specialist_generate, llm_name)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VLM (Qwen3-VL 등) + 프롬프트                                                 │
│  "List all distinct objects visible in this image. Output comma-separated."  │
│  → image + prompt → "chair, dining table, person, trash can, bottle, ..."    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    object_names = ["chair", "dining table", "person", "trash can", ...]
```

**고정 목록 없음** — VLM이 이미지를 보고 동적으로 객체를 나열.

---

### 6.4 Step 2b: get_3d_representation(image, object_names) 내부 동작

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  get_3d_representation(image, object_names)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Sub-Tool 1: get_depth_map    │         │  Sub-Tool 2: get_detections_   │
│  (depth.py)                   │         │  with_labels (open_vocab)     │
│                               │         │                               │
│  • DepthAnythingV2-Small      │         │  • OWL-ViT (object_names로)   │
│  • image → depth map (H×W)     │         │  • object_names → bboxes      │
│  • lower value = closer       │         │  • Fallback: DETR (object_names │
└──────────────┬───────────────┘         │    없을 때)                    │
               │                         └──────────────┬─────────────────┘
               │                                        │
               │  depth_arr (numpy 2D)                   │  objects (list of dict)
               │                                        │
               └────────────────┬───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  조합 (representation_3d.py)                                                 │
│                                                                              │
│  for each object:                                                            │
│    • bbox 영역의 depth map mean → mean_depth                                  │
│    • bbox center → position (top-left, center, bottom-right 등)               │
│  objects를 mean_depth 기준 정렬 (낮을수록 closer)                              │
│  → Object Depth Ordering + Pairwise Relations 텍스트 생성                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tool 출력 예시 (텍스트)                                                      │
│                                                                              │
│  ## 3D Representation Tool Output (object-level depth)                       │
│                                                                              │
│  ### Object Depth Ordering (closer to camera → farther)                      │
│    1. chair (bottom-center) — CLOSEST                                        │
│    2. dining table (center)                                                   │
│    3. person (mid-left)                                                      │
│    4. window (top-center) — FARTHEST                                         │
│                                                                              │
│  ### Pairwise Depth Relations (A in front of B = A closer to camera)        │
│    - chair is IN FRONT OF dining table                                       │
│    ...                                                                       │
│                                                                              │
│  ### Instance Count by Object Type                                           │
│    - chair: 2                                                                 │
│    - dining table: 1                                                          │
│    - person: 1                                                                │
│                                                                              │
│  ### How to Use                                                              │
│    - For 'how many X?': check Instance Count. For depth: use Ordering.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 6.5 Step 3: Prompt 구성 (build_role_prompt)

```python
# prompts.py - build_role_prompt(role="explicit_3d_representation", query, tool_output)
tool_section = "## Tool Output (use this data in your reasoning)\n\n" + tool_output
role_prompt = template.format(query=query, tool_section=tool_section)
```

**최종 prompt 구조:**
```
# ROLE: Explicit 3D Representation Construction Agent
You are a **3D depth reasoning specialist**...

## Tool Output (use this data in your reasoning)

[위 6.3의 Tool 출력 전체]

## How to Use the 3D Representation Tool Output
...

## Task
Question: {query}   ← 벤치마크 쿼리 (질문 + Options)

## Output Format (STRICT)
...
```

---

### 6.6 Step 4: VLM 추론 (specialist_generate)

```python
raw_output = specialist_generate(llm_name, image, role_prompt)
```

**VLM에 전달되는 입력:**
| 입력 | 내용 |
|------|------|
| **image** | 벤치마크 원본 이미지 (PIL) |
| **role_prompt** | ROLE 설명 + Tool 출력 + query + 출력 형식 |

**VLM이 하는 일:**
1. 이미지를 시각적으로 이해
2. Tool 출력에서 질문 관련 객체 찾기 (semantic match)
3. Object Depth Ordering / Pairwise Relations로 depth 비교
4. `Answer: (X)` + `Reason: ...` 형식으로 출력

---

### 6.7 Step 5: 출력 파싱 → SharedMemory

```python
answer, reason = parse_specialist_output(raw_output)
shared_memory.add("explicit_3d_representation", llm_name, answer, reason)
```

---

### 6.8 explicit_3d 전체 데이터 흐름 요약

```
image + query (벤치마크)
    │
    ├── image ──► extract_objects_from_image(image, specialist_generate, llm_name)
    │                    │
    │                    └── VLM + "List all objects" prompt → object_names
    │
    ├── image + object_names ──► get_3d_representation(image, object_names)
    │                    │
    │                    ├── get_depth_map(image)     → DepthAnything
    │                    ├── get_detections_with_labels(image, object_names) → OWL-ViT
    │                    └── 조합 → tool_output (텍스트, Instance Count 포함)
    │
    └── query ──► build_role_prompt(role, query, tool_output)
                           │
                           └── role_prompt = ROLE + tool_output + query + format
                                    │
                                    ▼
                    specialist_generate(llm, image, role_prompt)
                                    │
                                    ▼
                    Answer: (X) + Reason: ...
```

---

## 7. 관련 파일

| 파일 | 역할 |
|------|------|
| `src2/benchmarks/loaders.py` | 벤치마크 로드, get_benchmark_* |
| `src2/agents/mas_v2/pipeline.py` | run_step, run_train, run_test |
| `src2/agents/mas_v2/prompts.py` | build_head_agent_prompt, build_role_prompt, build_final_reasoning_prompt |
| `src2/agents/mas_v2/score_map.py` | select_agents |
| `src2/agents/mas_v2/shared_memory.py` | specialist 출력 수집 |
| `src2/tools/representation_3d.py` | get_3d_representation(image, object_names) — 메인 |
| `src2/tools/object_extraction.py` | extract_objects_from_image — VLM으로 객체 목록 추출 |
| `src2/tools/open_vocab_detection.py` | get_detections_with_labels — OWL-ViT |
| `src2/tools/depth.py` | get_depth_map(image) — DepthAnything |
| `src2/tools/scene_graph.py` | get_detected_objects(image) — DETR (fallback) |
| `run_eval_mas_v2.py` | 진입점, build_runners, run_experiment |
