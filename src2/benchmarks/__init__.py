"""
Benchmark loaders for Spatial Reasoning evaluation.
- CV-Bench, 3DSRBench
"""
from .loaders import (
    load_benchmark,
    BENCHMARK_CONFIGS,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_category,
    is_multiple_choice,
    infer_answer_type_from_query,
    SPATIAL_TASK_CATEGORIES,
)

__all__ = [
    "load_benchmark",
    "BENCHMARK_CONFIGS",
    "get_benchmark_prompt",
    "get_benchmark_answer",
    "get_benchmark_image",
    "get_benchmark_category",
    "is_multiple_choice",
    "infer_answer_type_from_query",
    "SPATIAL_TASK_CATEGORIES",
]
