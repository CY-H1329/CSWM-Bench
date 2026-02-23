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
# Unified spatial category taxonomy — 5 categories
#
# Grounded in cognitive neuroscience:
#   1. spatial_relation  — Categorical spatial processing (left parietal)
#   2. distance_depth    — Coordinate spatial processing (right parietal)
#   3. size              — Magnitude processing (IPS)
#   4. orientation       — Mental rotation / direction (parietal)
#   5. counting          — Numerosity processing (IPS)
#
# References:
#   - Kosslyn 1987: categorical vs coordinate spatial representations
#   - Walsh 2003 (ATOM): shared magnitude system for size/number
#   - Levinson 2003: frames of reference (intrinsic/relative/absolute)
#   - SpatialBench, VSI-Bench, SpatialRGPT-Bench: benchmark convergence
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    "spatial_relation",
    "distance_depth",
    "size",
    "orientation",
    "counting",
]

CATEGORY_DESCRIPTIONS = {
    "spatial_relation":
        "Positional relationship between objects: above/below, left/right, "
        "in front of/behind, next to, between, inside/outside. "
        "Asks WHERE one object is relative to another.",
    "distance_depth":
        "How far apart objects are (allocentric distance) or how far an object "
        "is from the camera/viewer (egocentric depth). "
        "Asks HOW FAR — includes both depth and distance questions.",
    "size":
        "Comparing the size, height, or scale of objects: taller/shorter, "
        "bigger/smaller, wider/narrower. "
        "Asks HOW BIG one object is compared to another.",
    "orientation":
        "Which direction objects face, viewpoint-dependent questions, rotation, "
        "parallel/perpendicular arrangement, same/different facing direction. "
        "Asks WHICH WAY objects are oriented.",
    "counting":
        "Counting how many objects or instances of a type exist in the scene. "
        "Asks HOW MANY.",
}

# ---------------------------------------------------------------------------
# Mapping: fine-grained benchmark categories → 5 unified categories
# Used for GT evaluation (never passed to Head Agent)
# ---------------------------------------------------------------------------
FINE_TO_UNIFIED = {
    # 3DSRBench (12 categories)
    "location_above": "spatial_relation",
    "height_higher": "size",
    "location_closer_to_camera": "distance_depth",
    "multi_object_closer_to": "distance_depth",
    "location_next_to": "spatial_relation",
    "orientation_on_the_left": "spatial_relation",
    "orientation_in_front_of": "spatial_relation",
    "orientation_viewpoint": "orientation",
    "multi_object_facing": "orientation",
    "multi_object_same_direction": "orientation",
    "multi_object_viewpoint_towards_object": "orientation",
    "multi_object_parallel": "orientation",
    # CV-Bench (4 categories)
    "Count": "counting",
    "Relation": "spatial_relation",
    "Depth": "distance_depth",
    "Distance": "distance_depth",
}

# ---------------------------------------------------------------------------
# Score map defaults
# ---------------------------------------------------------------------------
INITIAL_SCORE = 0.5
DEFAULT_SCORE_DELTA_CORRECT = 0.05
DEFAULT_SCORE_DELTA_WRONG = -0.02
