# Baseline Experiments

This document describes the baseline experiments and how they were conducted.

## Experiment types

### 1. Single-agent (3DSRBench)

One model per sample. Direct accuracy comparison.

**Scripts**:
- `run_eval_single_3dsrbench.py` — All GPU models in one run
- `scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py` — Qwen3 only (recommended)
- `scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py` — Sa2VA only
- `scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py` — LLaVA4D only

**Commands**:
```bash
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
```

### 2. API models (3DSRBench)

Claude, GPT-4o, Gemini on 3DSRBench (100 or full dataset).

**Script**: `scripts/evals/3dsrbench_api/run_eval_api.py`

**Commands**:
```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
```

### 3. MAS pipeline

Head → Perception → Reasoning. One model combination per run.

**Script**: `run_eval_mas.py`

**Commands**:
```bash
python run_eval_mas.py --benchmark 3dsrbench --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b
python run_eval_mas.py --benchmark cvbench --head sa2va --perception sa2va --reasoning sa2va
```

### 4. MAS full

All model combinations (Qwen3, Sa2VA, LLaVA4D × 3) on a benchmark.

**Script**: `run_eval_mas_full.py`

**Commands**:
```bash
python run_eval_mas_full.py --benchmark 3dsrbench
```

## Per-dataset summary

| Dataset | GPU models | API models | MAS |
|---------|------------|------------|-----|
| 3DSRBench | ✓ (Qwen3, Sa2VA, LLaVA4D) | ✓ (Claude, GPT-4o, Gemini) | ✓ |
| CV-Bench | ✓ (Qwen3, Sa2VA, LLaVA4D) | — | ✓ |

## Per-model summary

| Model | 3DSRBench | CV-Bench |
|-------|-----------|----------|
| Qwen3-4B | GPU + MAS | GPU + MAS |
| Sa2VA-4B | GPU + MAS | GPU + MAS |
| LLaVA4D | GPU + MAS | GPU + MAS |
| Claude Sonnet 4.5 | API | — | — |
| GPT-4o | API | — | — |
| Gemini | API | — | — |

## Aggregation

Category-level performance for 3DSRBench:

```bash
python scripts/evals/3dsrbench/aggregate_category_performance.py \
  --dir results/runs/3dsrbench/api_models/20260216_121420 \
  --runs_file scripts/evals/3dsrbench/runs_api_claude.json \
  --output results/runs/3dsrbench/api_models/20260216_121420/category_claude.csv
```
