# Methodology

This document describes the methodology used in Spatial_MAS for evaluating spatial reasoning in vision-language models.

## Overview

Spatial_MAS evaluates multimodal models on spatial reasoning benchmarks through:

1. **Single-agent baseline**: One model per sample, direct comparison (3DSRBench, CV-Bench).
3. **MAS pipeline**: Head → Perception → Reasoning pipeline with specialized agents.

## Benchmarks

| Benchmark | Description | Split | Options |
|-----------|-------------|-------|---------|
| **3DSRBench** | 3D spatial reasoning (ccvl/3DSRBench) | test | A/B/C/D |
| **CV-Bench** | Computer vision reasoning (nyu-visionx/CV-Bench) | test | Multiple choice |

## Models

### GPU models (local inference)

- **Qwen3-VL-4B**: Qwen/Qwen3-VL-4B-Instruct
- **Sa2VA-4B**: ByteDance/Sa2VA-4B
- **LLaVA4D**: LLaVA-NeXT 7B (fallback until LLaVA-4D release)

### API models (cloud)

- **Claude Sonnet 4.5**: Anthropic
- **GPT-4o**: OpenAI
- **Gemini Robotics-ER**: Google
- **DeepSeek-VL**: DeepSeek (optional)

## Evaluation protocol

- **Temperature**: 0.0 (greedy) for single-agent; 0.4 for multi-agent/collaboration.
- **Answer extraction**: Normalized to A/B/C/D for multiple-choice benchmarks.
- **Metrics**: Accuracy, per-category accuracy (when categories available).

## MAS pipeline

The Multi-Agent System (MAS) pipeline consists of:

1. **Head Agent**: Task classification (depth, distance, relation, etc.).
2. **Perception Agent**: Information extraction from the image.
3. **Reasoning Agent**: Spatial reasoning and final answer.

Prompts are defined in `src/agents/prompts.yaml`.
