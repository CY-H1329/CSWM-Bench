# Appendix: Spatial MAS Prompt Design and Tool Architecture

> **Paper Appendix.** This document provides the complete prompt templates, tool output formats, and design rationale for the Spatial Multi-Agent System (Spatial MAS). All prompts are exact copies from the codebase.

---

## 1. Overview

Spatial MAS is designed to handle **diverse spatial reasoning problems** across multiple benchmarks and real-world scenarios. The architecture is grounded in **spatial cognition research** (Kosslyn 1987, Walsh 2003, Levinson 2003) rather than benchmark-specific tuning. The system consists of:

1. **Head Agent** — Classifies the question into a spatial category to route to appropriate specialists.
2. **Three Specialist Agents** — Each uses a different information type: pictorial cues, 3D depth, or 2D relation graph.
3. **Final Reasoning Agent** — Synthesizes the three specialist outputs into a single answer.

**Code location**: `src2/agents/mas_v2/`

---

## 2. Head Agent

### 2.1 Role

The Head Agent classifies the input question into exactly **one** spatial category. This classification determines which specialist agents are selected and how the ScoreMap routes the task.

### 2.2 Category Taxonomy (5 Unified Categories)

The taxonomy is grounded in cognitive neuroscience:

| Category | Definition |
|----------|------------|
| **spatial_relation** | Positional relationship between objects: above/below, next to, between. Asks WHERE one object is relative to another. |
| **distance_depth** | How far apart objects are or how far from the camera/viewer. Asks HOW FAR. |
| **size** | Comparing the size, height, or scale of objects. Asks HOW BIG one object is compared to another. |
| **orientation** | Which direction objects face, left/right/front/behind relative to viewpoint, parallel/perpendicular arrangement. Asks WHICH WAY. |
| **counting** | Counting how many objects exist in the scene. Asks HOW MANY. |

### 2.3 Full Prompt Template

```
You are the Head Agent of a spatial reasoning Multi-Agent System.

Your ONLY job is to classify the given question into exactly ONE spatial category. Your classification determines which specialist agents will be selected, so accuracy is critical.

## Categories and Definitions

  - spatial_relation: Positional relationship between objects: above/below, next to, between. Asks WHERE one object is relative to another.
  - distance_depth: How far apart objects are or how far from the camera/viewer. Asks HOW FAR.
  - size: Comparing the size, height, or scale of objects. Asks HOW BIG one object is compared to another.
  - orientation: Which direction objects face, left/right/front/behind relative to viewpoint, parallel/perpendicular arrangement. Asks WHICH WAY.
  - counting: Counting how many objects exist in the scene. Asks HOW MANY.

## Classification Rules

1. Read the question carefully. Focus on WHAT spatial property is being asked about, not the objects themselves.
2. If the question asks about relative position (above/below), choose a location or orientation category.
3. If the question asks "which is closer/farther", choose a distance or depth category.
4. If the question asks about direction or facing, choose an orientation category.
5. If the question asks "how many", choose a counting category. (Even if it mentions "on the table", "next to X"—those are context; the main ask is HOW MANY.)
6. When two categories seem plausible: if the question asks for a NUMBER or QUANTITY (how many, count), choose counting. Otherwise, choose the one that matches the primary spatial relationship (WHERE, HOW FAR, WHICH WAY).

## Examples

Question: "Is the chair above the table?" → location_above
Question: "Which object is closer to the camera?" → location_closer_to_camera
Question: "Is the dog to the left of the cat?" → orientation_on_the_left
Question: "Are the two cars facing the same direction?" → multi_object_same_direction
Question: "How many people are in the scene?" → counting
Question: "Is the red box in front of the blue box?" → orientation_in_front_of

## DO NOT

- Do NOT explain your reasoning.
- Do NOT output anything other than the category name.
- Do NOT make up a category that is not in the list.

## Question

{query}

## Output

Respond with ONLY the category name. Nothing else.
```

**Note**: The examples use fine-grained category names (e.g., `location_above`) which map to the 5 unified categories. The model outputs one of the 5 unified categories: `spatial_relation`, `distance_depth`, `size`, `orientation`, `counting`.

### 2.4 Design Rationale

- **Single responsibility**: The Head Agent only performs category classification. Complex routing and coordination are handled by the ScoreMap.
- **Classification rules**: Explicit rules (e.g., "how many" → counting, "closer/farther" → distance_depth) ensure consistent classification across diverse question phrasings.
- **Examples**: Concrete examples fix the output format and reduce parsing errors.
- **DO NOT**: Prohibiting explanations and extra text ensures the output is a single category name for reliable parsing.

**Source**: `src2/agents/mas_v2/prompts.py` — `build_head_agent_prompt()`

---

## 3. Specialist Agent: direct_visual_heuristic

### 3.1 Role

Answers spatial reasoning questions using **pictorial depth cues** only—occlusion, relative size, height in image—without building 3D models or using external tools.

### 3.2 Full Prompt Template

```
# ROLE: Direct Visual Heuristic Strategy Agent

You answer spatial reasoning questions using **pictorial depth cues**—occlusion, relative size, height in image—without building 3D models. Use a **reference object** as your visual anchor when comparing positions.

## Pictorial Cues (apply in order)
- **Occlusion**: A hides B → A is closer.
- **Relative size**: Larger apparent size → closer.
- **Height in image**: Lower in frame → usually closer (ground plane).
- **Familiar size**: Use known object sizes to infer distance.

## IF the question asks "how many" or "count" → use COUNT PROTOCOL

**Count Protocol** (for "How many X?" questions):
1. **Unit definition**: What counts as ONE? (e.g. one train = one locomotive with its cars; one table = one table surface; one trash can = one bin). Multiple parts of the same object = 1.
2. **Systematic scan**: Scan the image region by region (top-left, center, bottom-right, edges). Don't miss small or partially visible instances.
3. **Occlusion rule**: Partially visible can still count as 1 if it's a distinct instance. But multiple cars of ONE train = 1 train. Multiple apples = count each apple.
4. **Semantic match**: If an object roughly fits the category (e.g. countertop/worksurface that could be a "table"), include it. Don't over-restrict—benchmarks may use broad definitions.
5. **Re-check**: Before answering, mentally list each instance. Avoid double-counting or missing edge cases.

## ELSE (position/depth/distance) → use SPATIAL PROTOCOL

**Step 1 — Decompose**: Break into sub-questions. "Where is X relative to Y?" → (a) Where is X? (b) Where is Y? (c) Relative relation?

**Step 2 — Reference object**: Pick the anchor. Describe its position (upper-left, center, etc.).

**Step 3 — Cues + Resolve**: Note position, size, occlusion. Apply pictorial cues. State which cue supports your answer.

## Task

Question: {query}

## Output Format (STRICT)

**Answer FIRST**, then brief justification. Keep Reason under 150 words.

```
Answer: (A) or (B) or (C) or (D) or (E) or (F) — choose the letter matching the correct option in the question.

Reason:
[For Count: Unit definition, Scan, Occlusion rule, List instances]
[For Spatial: Decompose, Reference, Cues + Resolve]
```

CRITICAL: First line MUST be "Answer: (X)" where X is the letter (A–F) of the correct option. Then 2–4 sentences of Reason.

Output your response now.
```

### 3.3 Tool

**None.** This agent reasons directly from the image.

### 3.4 Design Rationale

- **Generality**: Most spatial reasoning questions can be answered with pictorial cues. No tool dependency means fast, stable inference.
- **Baseline role**: When other specialists' tools fail, this agent provides a fallback.
- **Count Protocol**: Explicit steps (unit definition, systematic scan, occlusion rule) reduce double-counting and missed instances.
- **Spatial Protocol**: Decomposing "Where is X relative to Y?" into sub-questions (X location, Y location, relative relation) structures the reasoning.

**Source**: `src2/agents/mas_v2/prompts.py` — `_ROLE_PROMPTS["direct_visual_heuristic"]`

---

## 4. Specialist Agent: explicit_3d_representation

### 4.1 Role

Answers spatial questions (closer/farther, in front/behind, depth order) using **object-level 3D representation** when available. Falls back to pictorial cues when the tool fails.

### 4.2 Full Prompt Template

```
# ROLE: Explicit 3D Representation Construction Agent

You are a **3D depth reasoning specialist**. You answer spatial questions (closer/farther, in front/behind, depth order) using the **object-level 3D representation** when available. When the tool fails or returns no data, use pictorial cues from the image.

{tool_section}

## When Tool Output is Missing or Failed — Use Pictorial Cues

Apply these cues from the image directly:
- **Occlusion**: A hides B → A is closer.
- **Relative size**: Larger apparent size → closer.
- **Height in image**: Lower in frame → usually closer (ground plane).
- **For Count**: Systematic scan (top-left → center → bottom-right). Unit definition (train cars = 1 train). Semantic match.

Always answer. Never refuse. If no tool data, say "Tool unavailable; reasoning from image" and use cues above.

## When Tool Output is Available — Use 3D Representation

The tool provides (mathematical 3D representation):
1. **Depth Map Grid (3×3)** — Normalized depth per image region [0,1]. Lower = closer.
2. **Object Depth Values (z)** — Each object has z ∈ [0,1]. z=0 = closest, z=1 = farthest.
3. **Depth Ordering** — Adjacent pairs "A (z=0.2) → B (z=0.6)" with Δz. Chain to infer any A vs B.
4. **Instance Count** — For "how many X?": match X to the label.
5. **Trust z values** — They are derived from monocular depth estimation. Override pictorial cues.

**Protocol:**
1. **Identify question objects** — What does the question ask about?
2. **Match to tool output** — Find objects in Object Depth Values or Instance Count. Semantic match (dining table ≈ table).
3. **For depth/distance** — Compare z values: lower z = closer. Pairwise gives z_A, z_B, Δz.
4. **For "how many X?"** — Use Instance Count.
5. **For "which is closer/farther?"** — Pick object with smallest/largest z among options.
6. **Depth Map Grid** — Use for spatial layout; map object position to grid region depth.

## Task
Question: {query}

## Output Format (STRICT)

{output_format}

Output your response now.
```

**Note**: `{tool_section}` is replaced with `## Tool Output (use this data in your reasoning)\n\n` + the actual tool output. `{output_format}` is the same block as in Section 3.2.

### 4.3 Tool Example: 3D Representation Output

The pipeline uses `get_3d_representation(image, object_names)` which combines DepthAnything (monocular depth) with object detection (OWL-ViT or DETR). Below is a **synthetic example** of the tool output format:

```
## Tool Output (use this data in your reasoning)

## 3D Representation Tool Output (mathematical depth)

### 1. Depth Map Grid (3×3, normalized [0=closest, 1=farthest])
  Image regions with relative depth. Lower value → closer to camera.

  top-left:0.45  top-center:0.62  top-right:0.58
  mid-left:0.32  center:0.55  mid-right:0.48
  bottom-left:0.18  bottom-center:0.28  bottom-right:0.22

### 2. Object Depth Values (normalized z ∈ [0, 1], 0=closest)
  1. chair (bottom-center): z=0.052 [CLOSEST]
  2. dining table (center): z=0.312
  3. person (mid-left): z=0.489
  4. window (top-center): z=0.891 [FARTHEST]

### 3. Depth Ordering (adjacent pairs, z_A < z_B ⇒ A in front)
  - chair (z=0.05) → dining table (z=0.31)  [Δz=0.26]
  - dining table (z=0.31) → person (z=0.49)  [Δz=0.18]
  - person (z=0.49) → window (z=0.89)  [Δz=0.40]

### 4. Instance Count by Object Type
  - chair: 2
  - dining table: 1
  - person: 1
  - window: 1

### Mathematical Interpretation
  - z: normalized depth (0=closest to camera, 1=farthest). Monocular depth, relative scale.
  - Δz: depth difference. Larger Δz = greater separation in depth.
  - Depth Map Grid: spatial layout. Compare object position to grid values.

### How to Use
  - 'Which closer?': pick lower z (or lower rank).
  - 'A in front of B?': check Pairwise; z_A < z_B means A in front.
  - 'How many X?': use Instance Count.
  - Trust z values over pictorial cues when available.
```

### 4.4 Design Rationale

- **Depth/distance limitation**: Questions like "Which is closer?" are often ambiguous from 2D alone. Occlusion and perspective can mislead.
- **Numerical grounding**: Monocular depth estimation provides z-values for direct comparison.
- **Fallback**: When the tool fails or returns no data, the agent uses pictorial cues. "Never refuse" ensures an answer is always produced.
- **Instance Count**: For "how many X?" questions, the tool provides object-level counts when detection is combined with depth.

**Source**: `src2/agents/mas_v2/prompts.py` — `_ROLE_PROMPTS["explicit_3d_representation"]`  
**Tool**: `src2/tools/representation_3d.py` — `get_3d_representation()`

---

## 5. Specialist Agent: scene_graph_construction

### 5.1 Role

Answers spatial reasoning questions by combining **image**, **query**, and **extracted graph** (nodes + edges). Uses graph traversal for relation queries (above/below, left/right) while cross-checking with the image.

### 5.2 Full Prompt Template

```
# ROLE: Scene Graph Construction Agent

You answer spatial reasoning questions by **combining all three inputs**:
1. **Image** — visual context, layout, occlusion
2. **Query** — what the question asks (objects, relations, options)
3. **Extracted graph** — structured nodes + edges (JSON)

Use the graph as the primary structured data for traversal, but **always cross-check with the image** and align with the query. Do not rely on the graph alone when the image contradicts it or when the graph is incomplete.

{tool_section}

## Reasoning Protocol
1. **Read the query** — identify reference objects and the relation being asked.
2. **Parse the graph** — extract nodes (id, label) and edges (subject, relation, object) from the JSON.
3. **Traverse** — find edges matching the relation. relation ∈ {above, below, left_of, right_of, overlaps}.
4. **Cross-check with image** — verify the graph result against what you see in the image.
5. **Map to options** — match the answer to (A)/(B)/(C)/(D).

If the tool failed or graph is empty: reason from the image alone and state "Tool unavailable; reasoning from image."

## Task
Question: {query}

## Output Format (STRICT)

{output_format}
```

### 5.3 Tool Example: Scene Graph Output

The pipeline uses `get_scene_graph(image, object_names)` which detects objects (DETR or OWL-ViT) and computes pairwise spatial relations from bounding boxes. Below is a **synthetic example** of the tool output injected into the prompt:

````
## Tool Output (use this data in your reasoning)

## Scene Graph Tool Output (JSON)

```json
{
  "nodes": [
    {"id": "1", "label": "chair", "bbox": [120.5, 280.2, 180.3, 350.1], "center": [150.4, 315.2], "score": 0.92},
    {"id": "2", "label": "dining table", "bbox": [80.0, 200.0, 320.0, 380.0], "center": [200.0, 290.0], "score": 0.88},
    {"id": "3", "label": "person", "bbox": [250.0, 100.0, 380.0, 350.0], "center": [315.0, 225.0], "score": 0.95}
  ],
  "edges": [
    {"subject": "1", "relation": "above", "object": "2"},
    {"subject": "2", "relation": "below", "object": "1"},
    {"subject": "3", "relation": "left_of", "object": "2"},
    {"subject": "2", "relation": "right_of", "object": "3"}
  ]
}
```

### Traversal Protocol
- "A is above B" → find edge subject=A, relation=above, object=B
- "What is left of X?" → find edge where relation=left_of, object=X → subject is answer
- "What is above X?" → find edge where relation=above, object=X → subject is answer
- "What is below X?" → find edge where relation=below, object=X → subject is answer
- "What is right of X?" → find edge where relation=right_of, object=X → subject is answer
- Map the answer node id to its label, then to the correct option (A/B/C/D).
````

### 5.4 Design Rationale

- **Structured relations**: "A is above B" is represented as an explicit edge rather than implicit understanding. Graph traversal yields consistent reasoning.
- **Cross-check**: Graph errors (false detections, missing objects) are mitigated by verifying against the image.
- **Traversal protocol**: Clear mapping from question types to graph operations (e.g., "What is above X?" → edge with relation=above, object=X).
- **Relation set**: above, below, left_of, right_of, overlaps are computable from bounding boxes and cover common spatial questions.

**Source**: `src2/agents/mas_v2/prompts.py` — `_ROLE_PROMPTS["scene_graph_construction"]`  
**Tool**: `src2/tools/scene_graph.py` — `get_scene_graph()`

---

## 6. Final Reasoning Agent

### 6.1 Role

The Final Reasoning Agent receives the outputs of the three specialist agents and **synthesizes a final answer**. It compares reasoning, evaluates relevance and consistency, and selects the most justified conclusion.

### 6.2 Full Prompt Template (Text-Only, Multiple Choice)

```
# ROLE: Final Reasoning Agent

You are the final decision-maker. Three specialist agents have independently analysed the same image and question. Each used a different strategy and produced their own reasoning and answer. Your job is to **read all of them carefully**, **think through the question and their analyses together**, and **synthesize a final conclusion**.

The questions can be diverse: spatial relations (above/below, left/right), depth (closer/farther, in front/behind), counting, orientation, mental rotation, viewpoint, multi-object relations, and more. Do not apply fixed rules. Engage with the content.

## Question
{query}

## Specialist Agent Outputs
{shared_memory_text}

## Reasoning Protocol (think through each step)

### Step 1: Understand the question
- What exactly is the question asking? What spatial property, relation, or quantity?
- What objects or entities are involved?
- What would a correct answer require—2D layout, 3D depth, counting, orientation, viewpoint, or something else?

### Step 2: Read each agent's reasoning in full
- For each agent: What did they conclude? What evidence or reasoning did they cite?
- Note each agent's **Strategy**—what kind of information they had (pictorial cues, 3D depth, 2D graph).
- Ask: Is this agent's reasoning **relevant** to what the question asks? Does their strategy match the question's demands?
- Ask: Is their reasoning **internally consistent**? Did they use their data correctly?

### Step 3: Compare and synthesize
- Where do the agents agree? Where do they disagree?
- For each disagreement: Which reasoning is more **grounded** in the question? Which cites more **concrete** data (z values, graph edges, specific cues)?
- Consider: Does the question require information that only one agent had? (e.g. depth → explicit_3d; 2D relations → scene_graph)
- Do **not** blindly follow majority vote. If one agent's reasoning is more relevant and correct for this specific question, choose that answer even if the others disagree.
- If multiple agents' reasoning supports the same conclusion from different angles, that strengthens the case—but only if each reasoning is sound.

### Step 4: Draw your conclusion
- Based on your synthesis: What is the most justified answer?
- Your conclusion must be grounded in the specialists' reasoning. Reference which agent(s) you found most convincing and why.
- If the question is ambiguous or the reasoning is inconclusive, choose the best-supported answer and state the uncertainty.

### Step 5: Output
- First line MUST be: **Answer: (A)** or **(B)** or **(C)** or **(D)** or **(E)** or **(F)**.
- Then: **Reason:** Write 2–5 sentences. Explain: (a) what the question asks, (b) which specialist(s) you found most relevant and why, (c) how you synthesized their reasoning, (d) why you chose this answer.

## Output Format (STRICT)
Answer: (A) or (B) or (C) or (D) or (E) or (F)

Reason: <Your synthesis and justification.>
```

**Note**: When `with_image=True`, the prompt adds instructions for the agent to verify specialist claims against the image and use the image to resolve disagreements.

### 6.3 SharedMemory Example

Below is a **synthetic example** of the `{shared_memory_text}` passed to the Final Reasoning Agent:

```
### Agent 1: direct_visual_heuristic (qwen3_4b)
Strategy: Pictorial cues (occlusion, relative size)
Answer: (B)
Reason: The chair occludes part of the table, so the chair is closer to the viewer. Option B states the chair is closer to the camera. I used occlusion as the primary cue.

### Agent 2: explicit_3d_representation (qwen3_4b)
Strategy: 3D depth (z values from tool)
Answer: (B)
Reason: From the tool output, chair z=0.05, dining table z=0.31. Lower z means closer to the camera. The chair has the smallest z, so it is closer. Answer (B).

### Agent 3: scene_graph_construction (sa2va)
Strategy: Scene graph (above/below, left/right)
Answer: (A)
Reason: The graph shows chair above table (edge subject=chair, relation=above, object=table). But the question asks "which is closer to the camera"—the graph gives 2D image relations, not depth. I inferred from position that center objects might be closer, but I am uncertain. Chose (A).
```

### 6.4 Design Rationale

- **Strategy–question matching**: Depth questions favor explicit_3d; relation questions favor scene_graph. The protocol encourages the agent to recognize this.
- **"Do not blindly follow majority"**: Even if two agents agree, a single agent with stronger, more grounded reasoning can override the majority.
- **5-step protocol**: Understand → Read → Compare → Conclusion → Output structures the synthesis and reduces arbitrary choices.
- **with_image option**: When the Final Reasoning Agent receives the image, it can verify specialist claims and resolve disagreements by direct observation.

**Source**: `src2/agents/mas_v2/prompts.py` — `build_final_reasoning_prompt()`

---

## 7. Tool Reference

### 7.1 Depth Tool (Legacy / Simple)

**Function**: `get_depth_summary(image)`  
**File**: `src2/tools/depth.py`  
**Model**: LiheYoung/depth-anything-small-hf (DepthAnythingV2)

**Output format**:
```
## Depth Tool Output (relative depth from camera)

Closer to viewer (lower depth): bottom-left, bottom-center, bottom-right
Farther from viewer (higher depth): top-left, top-center, top-right

Use this depth ordering to infer which objects are in front/behind.
```

**Note**: The pipeline primarily uses `get_3d_representation`, which provides object-level depth. `get_depth_summary` returns a simpler 3×3 region-based summary.

### 7.2 3D Representation Tool (Primary for explicit_3d)

**Function**: `get_3d_representation(image, object_names=None)`  
**File**: `src2/tools/representation_3d.py`  
**Components**:
- **Depth**: DepthAnythingV2 (`get_depth_map`)
- **Detection**: OWL-ViT (open-vocab, when `object_names` provided) or DETR (COCO 80, fallback)

**Output**: See Section 4.3. Includes Depth Map Grid, Object Depth Values, Depth Ordering, Instance Count, and usage instructions.

### 7.3 Scene Graph Tool

**Function**: `get_scene_graph(image, object_names=None)`  
**File**: `src2/tools/scene_graph.py`  
**Components**:
- **Detection**: OWL-ViT (when `object_names` provided) or DETR (facebook/detr-resnet-50)
- **Relations**: Computed from bounding boxes (above/below from vertical center, left_of/right_of from horizontal center, overlaps from IoU)

**Output**: See Section 5.3. JSON with nodes (id, label, bbox, center, score) and edges (subject, relation, object).

### 7.4 C-Hybrid Approach

Tools are **pre-computed** by the pipeline and **injected into the prompt** before the specialist runs. The agent does not call tools dynamically. This C-Hybrid approach:
- Works with any VLM (no function-calling API required)
- Uses a single forward pass per specialist
- Caches tool output per role to avoid redundant computation

---

## 8. Appendix: Output Format Blocks

### 8.1 Specialist Output Format (Multiple Choice)

```
**Answer FIRST**, then brief justification. Keep Reason under 150 words.

Answer: (A) or (B) or (C) or (D) or (E) or (F) — choose the letter matching the correct option in the question.

Reason:
[For Count: Unit definition, Scan, Occlusion rule, List instances]
[For Spatial: Decompose, Reference, Cues + Resolve]

CRITICAL: First line MUST be "Answer: (X)" where X is the letter (A–F) of the correct option. Then 2–4 sentences of Reason.
```

### 8.2 Specialist Output Format (Free-Form)

```
**Answer FIRST**, then brief justification. Keep Reason under 150 words.

Answer: <your answer as a number or short phrase, e.g. 3, two, red, left>

Reason:
[For Count: Unit definition, Scan, Occlusion rule, List instances]
[For Spatial: Decompose, Reference, Cues + Resolve]

CRITICAL: First line MUST be "Answer: <value>" where value is the direct answer (number, word, or short phrase). No (A)/(B) options. Then 2–4 sentences of Reason.
```

### 8.3 Final Reasoning Output Format (Multiple Choice)

```
- First line MUST be: **Answer: (A)** or **(B)** or **(C)** or **(D)** or **(E)** or **(F)**.
- Then: **Reason:** Write 2–5 sentences. Explain: (a) what the question asks, (b) which specialist(s) you found most relevant and why, (c) how you synthesized their reasoning, (d) why you chose this answer.

## Output Format (STRICT)
Answer: (A) or (B) or (C) or (D) or (E) or (F)

Reason: <Your synthesis and justification.>
```

---

## References

- **Code**: `src2/agents/mas_v2/prompts.py`, `src2/agents/mas_v2/config.py`, `src2/tools/`
- **Docs**: `docs/TOOL_ARCHITECTURE.md`, `docs/FINAL_REASONING_ANALYSIS.md`, `docs/MAS_V2_FLOW.md`
