# Datasets

This document describes the benchmarks used for **spatial reasoning** evaluation, their characteristics, and why they were selected.

---

## Dataset Characteristics & Selection Rationale

### 3DSRBench

| Attribute | Value |
|-----------|-------|
| **HuggingFace** | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) |
| **Split** | test |
| **Samples** | ~5.1k |
| **Format** | Multiple choice (A/B/C/D) |
| **Categories** | 12 fine-grained (location, height, orientation, multi-object) |

**Why selected for spatial reasoning research:**

- **3D-centric**: Questions are explicitly designed for 3D spatial understanding (depth, occlusion, relative position, camera viewpoint).
- **Fine-grained categories**: 12 task types cover location (above, closer, next_to), height comparison, orientation (left, front, viewpoint), and multi-object relations (parallel, facing, same_direction).
- **Rigorous evaluation**: Multiple-choice format enables clear accuracy metrics; categories allow per-task analysis.
- **Complementary to 2D**: Focuses on 3D spatial reasoning that 2D benchmarks often miss.

---

### CV-Bench

| Attribute | Value |
|-----------|-------|
| **HuggingFace** | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| **Split** | test |
| **Samples** | ~2.6k |
| **Format** | Multiple choice (choices) |
| **Tasks** | Count, Relation (2D); Depth, Distance (3D) |
| **Sources** | ADE20K, COCO (2D); Omni3D (3D) |

**Why selected for spatial reasoning research:**

- **2D + 3D coverage**: Combines 2D spatial relationships & object counting (ADE20K, COCO) with 3D depth order & relative distance (Omni3D).
- **Vision-centric**: From Cambrian-1 project; probes fundamental visual understanding (spatial layout, occlusion, counting, depth, distance).
- **Diverse sources**: Repurposes standard vision benchmarks; natural language questions in multimodal context.
- **Complements 3DSRBench**: 3DSRBench is 3D-focused; CV-Bench adds 2D spatial tasks and broader vision reasoning.

---

## Summary

| Benchmark | 2D / 3D | Focus | Samples |
|-----------|---------|-------|---------|
| **3DSRBench** | 3D | Location, height, orientation, multi-object | ~5.1k |
| **CV-Bench** | 2D + 3D | Count, Relation, Depth, Distance | ~2.6k |

Together, 3DSRBench and CV-Bench provide a broad evaluation of spatial reasoning across 2D and 3D tasks.

---

## Usage

### 3DSRBench

```bash
# GPU
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset

# API
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
```

### CV-Bench

```bash
# GPU
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset

# API
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
```
