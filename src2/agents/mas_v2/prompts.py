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

    return f"""<role>
You are a spatial question classifier. You receive a question about a scene and output exactly ONE category name. Nothing else.
</role>

<categories>
{cats_block}
</categories>

<classification_priority>
Check categories in this EXACT order. Stop at the FIRST match.

<priority_1_counting>
TRIGGER: the question asks "how many", asks for a NUMBER of objects, or asks whether an object EXISTS.
OUTPUT: counting
EXAMPLES:
  "How many chairs are in the room?" → counting
  "Are there any dogs in the image?" → counting
  "Count the number of red objects." → counting
DO NOT confuse with other categories. If the question asks for a quantity, it is ALWAYS counting.
</priority_1_counting>

<priority_2_size>
TRIGGER: the question compares the PHYSICAL SIZE, HEIGHT, or SCALE of objects. Keywords: taller, shorter, bigger, smaller, larger, higher (when comparing object dimensions), wider, thinner, longest, tallest.
OUTPUT: size
EXAMPLES:
  "Which object is taller?" → size
  "Is the red building higher than the blue one?" → size
  "Which car is bigger?" → size
  "Is person A shorter than person B?" → size
CRITICAL: "higher" or "taller" comparing object dimensions = size, NOT spatial_relation.
  "Which is higher, the lamp or the shelf?" → size (comparing heights)
DO NOT classify size questions as spatial_relation. Size is about object dimensions, not positions.
</priority_2_size>

<priority_3_distance_depth>
TRIGGER: the question asks HOW FAR apart objects are, or which object is CLOSER/FARTHER from the camera, viewer, or another object. Keywords: closer, farther, nearer, distance, depth, how far, proximity.
OUTPUT: distance_depth
EXAMPLES:
  "Which object is closer to the camera?" → distance_depth
  "Is the car nearer to the viewer than the tree?" → distance_depth
  "How far apart are the two chairs?" → distance_depth
  "Which person is farther from the table?" → distance_depth
DO NOT classify distance/depth questions as spatial_relation. If "closer" or "farther" appears, it is distance_depth.
</priority_3_distance_depth>

<priority_4_orientation>
TRIGGER: the question asks about FACING DIRECTION, VIEWPOINT, ROTATION, or ALIGNMENT of objects. Keywords: facing, direction, oriented, viewpoint, parallel, perpendicular, looking at, pointing, turned, angle, same direction.
OUTPUT: orientation
EXAMPLES:
  "Are the two cars facing the same direction?" → orientation
  "Is the chair parallel to the wall?" → orientation
  "Which direction is the person looking?" → orientation
  "Is object A oriented towards object B?" → orientation
  "From which viewpoint is the scene captured?" → orientation
  "Are the bottles pointing the same way?" → orientation
DO NOT classify orientation questions as spatial_relation. If the question is about WHICH WAY something faces or points, it is orientation.
</priority_4_orientation>

<priority_5_spatial_relation>
TRIGGER: the question asks about the POSITIONAL RELATIONSHIP between objects — where one object is relative to another. Keywords: above, below, left, right, in front of, behind, next to, between, on top of, under, inside, outside, beside.
OUTPUT: spatial_relation
EXAMPLES:
  "Is the chair above the table?" → spatial_relation
  "Is the dog to the left of the cat?" → spatial_relation
  "Is the red box in front of the blue box?" → spatial_relation
  "Is object A next to object B?" → spatial_relation
spatial_relation is the DEFAULT only after all other categories have been ruled out.
</priority_5_spatial_relation>
</classification_priority>

<strict_output_rules>
- Output ONLY the category name. No explanation, no reasoning, no punctuation.
- NEVER answer the question itself. Only classify it.
- NEVER output a category not in the list.
- NEVER output multiple categories.
</strict_output_rules>

<question>
{query}
</question>"""


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
