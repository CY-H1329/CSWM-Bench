"""
Pipeline Head → Perception → Reasoning.
Head-Agent ne reçoit PAS la catégorie : il doit la classifier lui-même.
Prompts: src/agents/prompts.yaml
Docs: docs/experiments/baseline_experiments/single_agent/MAS_PROMPTS.md
"""
from pathlib import Path
from typing import Callable, Optional

import yaml
from PIL import Image

# Catégories que le Head doit inférer (ne jamais les donner en entrée)
TASK_CATEGORIES = [
    "depth", "distance", "relation", "existence", "count",
    "instance_location", "orientation", "size", "reach",
]

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


def _load_prompts() -> dict:
    if _PROMPTS_PATH.exists():
        with open(_PROMPTS_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


MAS_PIPELINE_PROMPTS = _load_prompts() or {
    "head": "Question: {query}\n\nClassify into ONE of: depth, distance, relation, existence, count, instance_location, orientation, size, reach.\n\nReply with ONLY the category name.",
    "perception": "Task: {task_class}\nQuestion: {query}\n\nExtract relevant information from the image. Provide a concise summary (3-6 sentences).",
    "reasoning": "Task: {task_class}\nQuestion: {query}\n\nExtracted: {perception_output}\n\nReason step by step. If multiple choice, reply 'Answer: (X)'. Otherwise give the direct answer.",
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
