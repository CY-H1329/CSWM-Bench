"""
Head-Agent Category Routing prompts.
Given a question (and optionally image), route to the closest benchmark category.
"""
from typing import List

# CV-Bench: 4 categories
CVBENCH_CATEGORIES = ["Count", "Relation", "Depth", "Distance"]

# 3DSRBench: 12 categories
DSR3BENCH_CATEGORIES = [
    "location_above",
    "height_higher",
    "location_closer_to_camera",
    "multi_object_closer_to",
    "orientation_on_the_left",
    "multi_object_facing",
    "multi_object_same_direction",
    "orientation_in_front_of",
    "multi_object_viewpoint_towards_object",
    "orientation_viewpoint",
    "location_next_to",
    "multi_object_parallel",
]


def get_categories(benchmark: str) -> List[str]:
    if benchmark == "cvbench":
        return CVBENCH_CATEGORIES
    if benchmark == "3dsrbench":
        return DSR3BENCH_CATEGORIES
    raise ValueError(f"Unknown benchmark: {benchmark}")


def build_category_routing_prompt(question: str, benchmark: str = "cvbench") -> str:
    """
    Build prompt for Head-Agent category routing.
    Model selects the ONE category closest to what the question is asking.
    """
    categories = get_categories(benchmark)
    cats = "\n".join(f"- {c}" for c in categories)
    return f"""# CATEGORY ROUTING (Head-Agent)

This benchmark ({benchmark.upper()}) uses the following task categories. Given the question below, select the ONE category that you think is **closest** to what the question is asking.

## Available categories

{cats}

## Instructions

1. Read the question carefully.
2. Consider which category best matches the spatial/visual reasoning required.
3. Select exactly ONE category — the one you think is the **closest match**.
4. If multiple seem relevant, choose the most dominant one.

## Output format

Reply with exactly:
Category: <your selected category>

Use the exact category name from the list above.

---

## Question

{question}
"""
