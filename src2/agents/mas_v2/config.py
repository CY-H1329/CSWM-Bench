"""
MAS v2 configuration.
Models, roles, categories, score defaults.
"""

# ---------------------------------------------------------------------------
# Specialist LLMs (open-source VLMs)
# ---------------------------------------------------------------------------
# Full set (6 agents, includes SpaceOm)
SPECIALIST_LLMS = [
    "qwen3_4b",
    "sa2va",
    "llava4d",
    "spatial_rgpt",
    "spaceom",
    "spatial_reasoner",
]

# Default TTO specialists (4 VLMs). SpatialRGPT is optional — requires cloning
# https://github.com/AnjieCheng/SpatialRGPT and SPATIALRGPT_PATH; use
# SPECIALIST_LLMS_5_WITH_RGPT or CLI --with_spatial_rgpt when the repo is installed.
# Excludes spaceom (optional, use --with_spaceom to add)
SPECIALIST_LLMS_5 = [
    "qwen3_4b",
    "sa2va",
    "llava4d",
    "spatial_reasoner",
]

SPECIALIST_LLMS_5_WITH_RGPT = [
    "qwen3_4b",
    "sa2va",
    "llava4d",
    "spatial_rgpt",
    "spatial_reasoner",
]

# Low-memory 3 agents (OOM / quick test)
SPECIALIST_LLMS_3 = [
    "qwen3_4b",
    "llava4d",
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
# Fine-grained benchmark categories (passed to Head Agent prompt)
# The model classifies into these concrete categories, then we map to unified.
# ---------------------------------------------------------------------------
CATEGORIES_3DSRBENCH = [
    "location_above", "height_higher",
    "location_closer_to_camera", "multi_object_closer_to",
    "location_next_to",
    "orientation_on_the_left", "orientation_in_front_of", "orientation_viewpoint",
    "multi_object_facing", "multi_object_same_direction",
    "multi_object_viewpoint_towards_object",
    "multi_object_parallel",
]

CATEGORIES_CVBENCH = ["Count", "Relation", "Depth", "Distance"]

ALL_FINE_CATEGORIES = CATEGORIES_3DSRBENCH + CATEGORIES_CVBENCH  # 16 total

FINE_CATEGORY_DESCRIPTIONS = {
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
        "Viewpoint-dependent orientation — whether objects are oriented towards a specific target.",
    "multi_object_parallel":
        "Whether multiple objects are arranged in parallel alignment.",
    "Count":
        "Counting the number of objects or instances of a specific type in the scene.",
    "Relation":
        "Spatial relationship between objects (left/right, above/below, inside/outside, etc.).",
    "Depth":
        "Relative depth ordering — which object is closer to or farther from the camera.",
    "Distance":
        "Estimating or comparing distances between objects in the scene.",
}

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
        "Positional relationship between objects: above/below, next to, between. "
        "Asks WHERE one object is relative to another.",
    "distance_depth":
        "How far apart objects are or how far from the camera/viewer. "
        "Asks HOW FAR.",
    "size":
        "Comparing the size, height, or scale of objects. "
        "Asks HOW BIG one object is compared to another.",
    "orientation":
        "Which direction objects face, left/right/front/behind relative to viewpoint, "
        "parallel/perpendicular arrangement. Asks WHICH WAY.",
    "counting":
        "Counting how many objects exist in the scene. Asks HOW MANY.",
}

# ---------------------------------------------------------------------------
# Mapping: fine-grained → 5 unified categories
#
# This mapping preserves the grouping that achieved 98.5% on 3DSRBench:
#   spatial_relation ← vertical + adjacency groups
#   distance_depth   ← camera_dist group
#   orientation      ← orientation + alignment groups
# ---------------------------------------------------------------------------
FINE_TO_UNIFIED = {
    # 3DSRBench → spatial_relation (vertical + adjacency)
    "location_above": "spatial_relation",
    "height_higher": "spatial_relation",
    "location_next_to": "spatial_relation",
    # 3DSRBench → distance_depth
    "location_closer_to_camera": "distance_depth",
    "multi_object_closer_to": "distance_depth",
    # 3DSRBench → orientation (orientation + alignment)
    "orientation_on_the_left": "orientation",
    "orientation_in_front_of": "orientation",
    "orientation_viewpoint": "orientation",
    "multi_object_facing": "orientation",
    "multi_object_same_direction": "orientation",
    "multi_object_viewpoint_towards_object": "orientation",
    "multi_object_parallel": "orientation",
    # CV-Bench
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
