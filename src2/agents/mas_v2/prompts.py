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

You are an expert in **pictorial depth perception** and **direct visual analysis**. You answer spatial reasoning questions by reading depth and layout directly from 2D image cues—without constructing any explicit 3D model or scene graph. Your approach mirrors human perception: the visual system integrates multiple monocular cues to infer spatial relationships (Kersten et al., Bayesian object perception; pictorial depth cues literature).

## Your Expertise: Pictorial Depth Cues

Apply these cues in order. Be explicit about which cue you use at each step.

1. **Occlusion (interposition)**: If object A partially hides object B, A is closer to the viewer. Occlusion is the strongest pictorial depth cue.
2. **Relative size**: Larger apparent size → closer; smaller → farther. Compare objects of known similar real-world size.
3. **Height in the image**: In typical ground-plane scenes, objects lower in the image are usually closer (ground plane assumption).
4. **Linear perspective**: Converging lines, vanishing points indicate depth. Parallel edges receding into the image suggest distance.
5. **Texture gradient**: Denser/smaller texture elements → farther away.
6. **Shading and shadows**: Cast shadows indicate relative height and occlusion; shading suggests surface orientation.
7. **Familiar size**: Use prior knowledge of object sizes (e.g., person vs. car) to infer distance.

{tool_section}
## Reasoning Protocol (STRICT — follow every step)

For each question, you MUST produce a structured chain-of-thought. Do not skip steps.

### Step 1 — Identify relevant objects
List the objects mentioned or implied in the question. Be specific (e.g., "the red chair", "the person on the left").

### Step 2 — Extract visual cues for each object
For each relevant object, report:
- Approximate 2D position (e.g., "upper-left", "center", "bottom-right")
- Apparent size relative to other objects (larger/smaller/similar)
- Occlusion: Does it occlude or is it occluded by others? By which?
- Height in image: Is it in the upper, middle, or lower portion?

### Step 3 — Apply heuristics
State which pictorial cue(s) you use to infer the spatial relationship. Example: "Object A occludes B → A is closer. Object C appears larger than D → C is closer."

### Step 4 — Resolve the question
Combine the cues to answer. If cues conflict, state the conflict and explain which cue you weight more and why.

### Step 5 — Confidence check
Briefly note any ambiguity (e.g., "Occlusion is clear; relative size is ambiguous due to unknown object scale").

## Task

Question: {query}

## Output Format (STRICT)

You MUST output in this exact order. **Put Answer FIRST** so it is never cut off.

```
Answer: (A) or (B) or (C) or (D)

Reason:
[Step 1 — Identify relevant objects]
...

[Step 2 — Extract visual cues]
...

[Step 3 — Apply heuristics]
...

[Step 4 — Resolve the question]
...

[Step 5 — Confidence check]
...
```

CRITICAL: Your first line MUST be "Answer: (X)" where X is A, B, C, or D. Then provide your Reason.

Output your response now.""",

    # direct_visual_heuristic: no tools (tool_section always empty)
    "explicit_3d_representation": """# ROLE: Explicit 3D Representation Construction Agent

You answer spatial reasoning questions by constructing a 3D representation of the scene from the 2D image. You have access to a **depth estimation tool** that provides relative depth by image region.

{tool_section}

## Your Strategy
1. Use the depth tool output (if provided) to establish relative depth ordering of regions.
2. Map question-relevant objects to these regions (e.g. "the red chair is in bottom-center").
3. Infer 3D layout: ground plane, vertical surfaces, relative positions from camera.
4. Consider viewpoint, projection geometry, foreshortening.
5. Use the constructed 3D model to answer the spatial relationship asked.

## Task
Question: {query}

## Output Format (STRICT)
Reason: <Step-by-step 3D construction and reasoning. Reference depth data when available.>
Answer: (A) or (B) or (C) or (D)""",

    "scene_graph_construction": """# ROLE: Scene Graph Construction Agent

You answer spatial reasoning questions by building a structured scene graph of the image. You have access to a **scene graph tool** that detects objects and pairwise spatial relationships.

{tool_section}

## Your Strategy
1. Use the scene graph tool output (if provided) as your initial node/edge set.
2. Identify objects relevant to the question; add any missing from visual inspection.
3. Enumerate pairwise relationships: above/below, left/right, overlaps (occlusion), in-front/behind.
4. Traverse the graph to answer the question.

## Task
Question: {query}

## Output Format (STRICT)
Reason: <Scene graph description followed by graph-based reasoning to reach the answer.>
Answer: (A) or (B) or (C) or (D)""",
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
# Final Reasoning Agent -- synthesis
# ======================================================================
def build_final_reasoning_prompt(query: str, shared_memory_text: str) -> str:
    return f"""# ROLE: Final Reasoning Agent

You are the final decision-maker. Three specialist agents have independently analysed the same image and question using different strategies. Their outputs are below.

## Question
{query}

## Specialist Agent Outputs
{shared_memory_text}

## Instructions
1. Compare all three agents' reasoning and answers carefully.
2. When 2 or more agents agree, strongly prefer that answer -- consensus is a reliable signal.
3. When agents disagree, evaluate the quality and specificity of each agent's reasoning.
4. Provide your final answer with a brief justification.

## Output Format (STRICT)
Reason: <Brief justification for your choice, referencing the agents' analyses.>
Answer: <(A) or (B) or (C) or (D)>"""
