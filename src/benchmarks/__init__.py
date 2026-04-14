"""
Benchmark loaders for Spatial Reasoning evaluation.
- CV-Bench, 3DSRBench, MMSI-Bench, …
"""
from .loaders import (
    load_benchmark,
    BENCHMARK_CONFIGS,
    get_benchmark_prompt,
    get_benchmark_answer,
    get_benchmark_image,
    get_benchmark_images,
    get_benchmark_category,
    SPATIAL_TASK_CATEGORIES,
    MMSI_BENCH_QUESTION_TYPES,
)

__all__ = [
    "load_benchmark",
    "BENCHMARK_CONFIGS",
    "get_benchmark_prompt",
    "get_benchmark_answer",
    "get_benchmark_image",
    "get_benchmark_images",
    "get_benchmark_category",
    "SPATIAL_TASK_CATEGORIES",
    "MMSI_BENCH_QUESTION_TYPES",
]
