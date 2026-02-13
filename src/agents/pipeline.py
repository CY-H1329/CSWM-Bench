"""
Pipeline Head → Perception → Reasoning.
Head-Agent ne reçoit PAS la catégorie : il doit la classifier lui-même.
"""
from typing import Callable, Optional

from PIL import Image

# Catégories que le Head doit inférer (ne jamais les donner en entrée)
TASK_CATEGORIES = [
    "depth", "distance", "relation", "existence", "count",
    "instance_location", "orientation", "size", "reach",
]

MAS_PIPELINE_PROMPTS = {
    "head": """Look at this image and the following question.

Question: {query}

What type of spatial reasoning task is this? Choose exactly ONE from:
depth, distance, relation, existence, count, instance_location, orientation, size, reach

Reply with ONLY the task name (one word or two words, e.g. "depth" or "instance_location").""",

    "perception": """You are the Perception Agent. You received:
- Task type (classified by Head): {task_class}
- Question: {query}

Describe what you observe or infer from the image that is relevant to answer this question.
You may mention: object locations, depths, sizes, spatial relationships, counts, etc.
Keep it concise (2-4 sentences).""",

    "reasoning": """You are the Reasoning Agent. You have:
- Task type: {task_class}
- Question: {query}
- Perception (from Perception Agent): {perception_output}

Based on this, provide the final answer to the question.
If the question has options (A), (B), (C), (D), reply with the correct letter, e.g. "Answer: (A)".
Otherwise, reply with the direct answer.""",
}


def _normalize_task_class(raw: str) -> str:
    """Extract task class from Head output."""
    raw = (raw or "").strip().lower()
    for c in TASK_CATEGORIES:
        if c in raw or c.replace("_", " ") in raw:
            return c
    return raw.split()[0] if raw else "relation"


def run_mas_pipeline(
    image: Image.Image,
    query: str,
    generate_fn: Callable[[Image.Image, str], str],
    head_model_fn: Optional[Callable[[Image.Image, str], str]] = None,
    perception_model_fn: Optional[Callable[[Image.Image, str], str]] = None,
    reasoning_model_fn: Optional[Callable[[Image.Image, str], str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict:
    """
    Run Head → Perception → Reasoning pipeline.
    Si head/perception/reasoning_model_fn sont None, on utilise generate_fn pour les 3.

    Returns:
        {
            "task_class": str,
            "perception_output": str,
            "final_answer": str,
        }
    """
    gen = head_model_fn or generate_fn
    perc = perception_model_fn or generate_fn
    reas = reasoning_model_fn or generate_fn

    # 1. Head-Agent: query + image ONLY (no category)
    head_prompt = MAS_PIPELINE_PROMPTS["head"].format(query=query)
    task_class_raw = gen(image, head_prompt)
    task_class = _normalize_task_class(task_class_raw)

    # 2. Perception Agent: query + task_class
    perc_prompt = MAS_PIPELINE_PROMPTS["perception"].format(
        task_class=task_class,
        query=query,
    )
    perception_output = perc(image, perc_prompt)

    # 3. Reasoning Agent: query + task_class + perception_output
    reas_prompt = MAS_PIPELINE_PROMPTS["reasoning"].format(
        task_class=task_class,
        query=query,
        perception_output=perception_output,
    )
    final_answer = reas(image, reas_prompt)

    return {
        "task_class": task_class,
        "perception_output": perception_output,
        "final_answer": final_answer,
    }
