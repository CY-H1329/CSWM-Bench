"""
MAS v2 prompts.

- Head Agent (Qwen3-VL-4B): category inference from image + query
- 3 Specialist Role prompts: Direct Visual Heuristic, Explicit 3D Repr, Scene Graph
- Final Reasoning Agent (DeepSeek-R1, text-only): synthesis from SharedMemory
"""
from typing import Dict, List, Optional


# ======================================================================
# Head Agent -- category inference (Qwen3-VL-4B, image + text)
# ======================================================================
def build_head_agent_prompt(
    query: str,
    category_list: List[str],
    category_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    if category_descriptions:
        cats_block = "\n".join(
            f"  - {c}: {category_descriptions.get(c, '')}" for c in category_list
        )
    else:
        cats_block = "\n".join(f"  - {c}" for c in category_list)

    return f"""You are the Head Agent of a spatial reasoning Multi-Agent System.

Your ONLY job is to classify the given question into exactly ONE spatial category. Your classification determines which specialist agents will be selected, so accuracy is critical.

## Categories and Definitions

{cats_block}

## Classification Rules

1. Read the question carefully. Focus on WHAT spatial property is being asked about, not the objects themselves.
2. If the question asks about relative position (above/below), choose a location or orientation category.
3. If the question asks "which is closer/farther", choose a distance or depth category.
4. If the question asks about direction or facing, choose an orientation category.
5. If the question asks "how many", choose a counting category.
6. When two categories seem plausible, choose the one that matches the CORE spatial relationship being asked.

## Examples

Question: "Is the chair above the table?" → location_above
Question: "Which object is closer to the camera?" → location_closer_to_camera
Question: "Is the dog to the left of the cat?" → orientation_on_the_left
Question: "Are the two cars facing the same direction?" → multi_object_same_direction
Question: "How many people are in the scene?" → Count
Question: "Is the red box in front of the blue box?" → orientation_in_front_of

## DO NOT

- Do NOT explain your reasoning.
- Do NOT output anything other than the category name.
- Do NOT make up a category that is not in the list.

## Question

{query}

## Output

Respond with ONLY the category name. Nothing else."""


# ======================================================================
# Specialist Role prompts
# ======================================================================
_ROLE_PROMPTS = {
    "direct_visual_heuristic": """# ROLE: Direct Visual Heuristic Strategy Agent

You answer spatial reasoning questions using **pictorial depth cues**—occlusion, relative size, height in image—without building 3D models. Use a **reference object** as your visual anchor when comparing positions.

## Pictorial Cues (apply in order)
- **Occlusion**: A hides B → A is closer.
- **Relative size**: Larger apparent size → closer.
- **Height in image**: Lower in frame → usually closer (ground plane).
- **Familiar size**: Use known object sizes to infer distance.

{tool_section}
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
Answer: (A) or (B) or (C) or (D)

Reason:
[For Count: Unit definition, Scan, Occlusion rule, List instances]
[For Spatial: Decompose, Reference, Cues + Resolve]
```

CRITICAL: First line MUST be "Answer: (X)" where X is A, B, C, or D. Then 2–4 sentences of Reason.

Output your response now.""",

    # direct_visual_heuristic: no tools (tool_section always empty)
    "explicit_3d_representation": """# ROLE: Explicit 3D Representation Construction Agent

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

**Answer FIRST**, then brief justification. Reference the 3D tool data explicitly.

```
Answer: (A) or (B) or (C) or (D)

Reason:
[1] Question objects: X, Y
[2] From tool: X rank N, Y rank M (or Pairwise: X in front of Y)
[3] Depth inference: X closer than Y (or vice versa)
[4] Conclusion → option (X)
```

CRITICAL: First line MUST be "Answer: (X)". If the tool failed, say so and reason from the image only.

Output your response now.""",

    "scene_graph_construction": """# ROLE: Scene Graph Construction Agent

You answer spatial reasoning questions by **combining all three inputs**:
1. **Image** — visual context, layout, occlusion
2. **Query** — what the question asks (objects, relations, options)
3. **Extracted graph** — structured nodes + edges (JSON)

Use the graph as the primary structured data for traversal, but **always cross-check with the image** and align with the query. Do not rely on the graph alone when the image contradicts it or when the graph is incomplete.

{tool_section}

## Reasoning Protocol
1. **Read the query** — identify reference objects and the relation being asked.
2. **Parse the graph** — extract nodes (id, label) and edges (subject, relation, object) from the JSON.
3. **Traverse** — find edges matching the relation. relation ∈ {{above, below, left_of, right_of, overlaps}}.
4. **Cross-check with image** — verify the graph result against what you see in the image.
5. **Map to options** — match the answer to (A)/(B)/(C)/(D).

If the tool failed or graph is empty: reason from the image alone and state "Tool unavailable; reasoning from image."

## Task
Question: {query}

## Output Format (STRICT)
Answer: (A) or (B) or (C) or (D)

Reason:
[1] From query: reference objects, relation asked
[2] From graph: edge(s) found
[3] From image: cross-check (if used)
[4] Conclusion → option (X)

CRITICAL: First line MUST be "Answer: (X)".""",
}


def build_role_prompt(role: str, query: str, tool_output: Optional[str] = None) -> str:
    """Build the specialist prompt for *role* with the given *query*.

    For explicit_3d_representation and scene_graph_construction, pass tool_output
    to inject depth or scene graph data (C hybrid approach).
    """
    template = _ROLE_PROMPTS.get(role)
    if template is None:
        raise ValueError(f"Unknown role: {role!r}. Choose from {list(_ROLE_PROMPTS)}")

    if tool_output and role in ("explicit_3d_representation", "scene_graph_construction"):
        tool_section = "## Tool Output (use this data in your reasoning)\n\n" + tool_output
    else:
        tool_section = ""

    return template.format(query=query, tool_section=tool_section)


# ======================================================================
# Final Reasoning Agent -- synthesis from SharedMemory
# ======================================================================
def build_final_reasoning_prompt(query: str, shared_memory_text: str) -> str:
    return f"""# ROLE: Final Reasoning Agent

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
- First line MUST be: **Answer: (A)** or **(B)** or **(C)** or **(D)**.
- Then: **Reason:** Write 2–5 sentences. Explain: (a) what the question asks, (b) which specialist(s) you found most relevant and why, (c) how you synthesized their reasoning, (d) why you chose this answer.

## Output Format (STRICT)
Answer: (A) or (B) or (C) or (D)

Reason: <Your synthesis and justification.>"""
