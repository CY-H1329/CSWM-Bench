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

1. Read the question carefully. Focus on WHAT spatial property is being asked about.
2. "above/below", "left/right", "in front of/behind", "next to", "between" → spatial_relation
3. "closer/farther", "how far", "depth", "distance", "nearer" → distance_depth
4. "taller/shorter", "bigger/smaller", "higher", "size" → size
5. "facing", "direction", "viewpoint", "parallel", "oriented" → orientation
6. "how many", counting objects → counting

## Examples

Question: "Is the chair above the table?" → spatial_relation
Question: "Is the dog to the left of the cat?" → spatial_relation
Question: "Which object is closer to the camera?" → distance_depth
Question: "How far apart are the two chairs?" → distance_depth
Question: "Which building is taller?" → size
Question: "Are the two cars facing the same direction?" → orientation
Question: "Is the bottle parallel to the wall?" → orientation
Question: "How many people are in the scene?" → counting

## DO NOT

- Do NOT explain your reasoning.
- Do NOT output anything other than the category name.
- Do NOT make up a category that is not in the list.
- Do NOT answer the question itself — only classify it.

## Question

{query}

## Output

Respond with ONLY the category name. Nothing else."""


# ======================================================================
# Specialist Role prompts
# ======================================================================
_ROLE_PROMPTS = {
    "direct_visual_heuristic": """# ROLE: Direct Visual Heuristic Strategy Agent

You answer spatial reasoning questions by directly analysing visual cues in the image WITHOUT constructing any intermediate representation.

## Your Strategy
1. Examine object positions, relative sizes, occlusion patterns, and perspective cues.
2. Use heuristics: objects lower in the image are typically closer; larger apparent size suggests proximity; occluding objects are in front.
3. Reason step-by-step from these visual observations.

## Task
Question: {query}

## Output Format (STRICT)
Reason: <Step-by-step visual reasoning. Be specific about what you observe.>
Answer: <(A) or (B) or (C) or (D)>""",

    "explicit_3d_representation": """# ROLE: Explicit 3D Representation Construction Agent

You answer spatial reasoning questions by mentally constructing a 3D representation of the scene from the 2D image.

## Your Strategy
1. Estimate the rough depth/distance of each relevant object from the camera.
2. Infer the 3D layout: ground plane, vertical surfaces, relative 3D positions.
3. Consider camera viewpoint, projection geometry, and foreshortening effects.
4. Use the constructed 3D model to reason about the spatial relationship asked.

## Task
Question: {query}

## Output Format (STRICT)
Reason: <Step-by-step 3D construction and reasoning. Describe estimated positions.>
Answer: <(A) or (B) or (C) or (D)>""",

    "scene_graph_construction": """# ROLE: Scene Graph Construction Agent

You answer spatial reasoning questions by building a structured scene graph of the image.

## Your Strategy
1. Identify all relevant objects in the scene with their attributes (position, size, orientation).
2. Enumerate pairwise spatial relationships: above/below, left/right, in-front/behind, near/far, facing direction.
3. Organise these into a graph structure (nodes = objects, edges = relationships).
4. Traverse the graph to answer the question.

## Task
Question: {query}

## Output Format (STRICT)
Reason: <Scene graph description followed by graph-based reasoning to reach the answer.>
Answer: <(A) or (B) or (C) or (D)>""",
}


def build_role_prompt(role: str, query: str) -> str:
    """Build the specialist prompt for *role* with the given *query*."""
    template = _ROLE_PROMPTS.get(role)
    if template is None:
        raise ValueError(f"Unknown role: {role!r}. Choose from {list(_ROLE_PROMPTS)}")
    return template.format(query=query)


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
