"""
MAS v2 configuration.
Models, roles, categories, score defaults.
"""

# ---------------------------------------------------------------------------
# 5 Specialist LLMs (open-source VLMs)
# ---------------------------------------------------------------------------
SPECIALIST_LLMS = [
    "qwen3_4b",
    "sa2va",
    "llava4d",
    "spatial_rgpt",
    "spatial_reasoner",
]

# ---------------------------------------------------------------------------
# 3 Specialist Roles
# ---------------------------------------------------------------------------
ROLES = [
    "direct_visual_heuristic",
    "explicit_3d_representation",
    "scene_graph_construction",
]

# ---------------------------------------------------------------------------
# Fixed models
# ---------------------------------------------------------------------------
HEAD_AGENT_MODEL = "qwen3_4b"           # Qwen3-VL-4B (VLM, image+text category inference)
REASONING_AGENT_MODEL = "deepseek_r1"   # DeepSeek-R1 (text-only, final reasoning)

# ---------------------------------------------------------------------------
# Unified spatial category taxonomy (16 categories, benchmark-agnostic)
#
# The Head Agent classifies ANY incoming question into the most fitting
# category from this fixed list.  The score map has one 5×3 sheet per
# category regardless of which benchmark the question came from.
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    # --- 3D spatial location ---
    "location_above",
    "height_higher",
    "location_closer_to_camera",
    "multi_object_closer_to",
    "location_next_to",
    # --- orientation / direction ---
    "orientation_on_the_left",
    "orientation_in_front_of",
    "orientation_viewpoint",
    "multi_object_facing",
    "multi_object_same_direction",
    "multi_object_viewpoint_towards_object",
    "multi_object_parallel",
    # --- counting / relation / depth / distance ---
    "count",
    "relation",
    "depth",
    "distance",
]

CATEGORY_DESCRIPTIONS = {
    # 3D spatial location
    "location_above":
        "Object A is above/below object B in 3D space (vertical positioning).",
    "height_higher":
        "Comparing the height or vertical extent of objects (which is taller/higher).",
    "location_closer_to_camera":
        "Which single object is closer to or farther from the camera viewpoint.",
    "multi_object_closer_to":
        "Among multiple objects, which one is closest to a reference object or point.",
    "location_next_to":
        "Whether two objects are adjacent or next to each other (proximity in horizontal plane).",
    # Orientation / direction
    "orientation_on_the_left":
        "Whether an object is on the left or right side relative to another object or the viewer.",
    "orientation_in_front_of":
        "Whether object A is in front of or behind object B in 3D space.",
    "orientation_viewpoint":
        "Questions about the camera or viewer's own orientation and perspective.",
    "multi_object_facing":
        "The facing direction of one or more objects (which way they are oriented).",
    "multi_object_same_direction":
        "Whether multiple objects face or point in the same direction.",
    "multi_object_viewpoint_towards_object":
        "Viewpoint-dependent orientation — whether objects are oriented towards a specific target from the viewer's perspective.",
    "multi_object_parallel":
        "Whether multiple objects are arranged in parallel alignment.",
    # Counting / relation / depth / distance
    "count":
        "Counting how many objects exist. The question explicitly asks 'how many' or a specific number.",
    "relation":
        "Spatial relationship between objects: left/right, above/below, in front of/behind, inside/outside, etc.",
    "depth":
        "Comparing which object is closer to or farther from the CAMERA or VIEWER. Keywords: 'closer to the camera', 'nearer', 'farther away', 'in front in the scene'. This is NOT about counting objects.",
    "distance":
        "Comparing or estimating the distance BETWEEN two objects (not relative to the camera). Keywords: 'how far apart', 'closer to object X', 'distance between'.",
}

# Legacy aliases (kept for backward compatibility if needed)
CATEGORIES_3DSRBENCH = ALL_CATEGORIES[:12]
CATEGORIES_CVBENCH = ALL_CATEGORIES[12:]

# ---------------------------------------------------------------------------
# Score map defaults
# ---------------------------------------------------------------------------
INITIAL_SCORE = 0.5
DEFAULT_SCORE_DELTA_CORRECT = 0.05
DEFAULT_SCORE_DELTA_WRONG = -0.02
