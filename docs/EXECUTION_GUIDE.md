# Execution Guide — Baseline Experiments

Single-agent baseline on **3DSRBench** and **CV-Bench**.  
See [BASELINE_EXPERIMENTS.md](BASELINE_EXPERIMENTS.md) for full commands.

---

## Overview

| Benchmark | GPU (Qwen3, Sa2VA, LLaVA4D) | API (Claude, GPT-4o, Gemini) |
|-----------|-----------------------------|------------------------------|
| 3DSRBench | `scripts/evals/3dsrbench/` | `scripts/evals/3dsrbench_api/` |
| CV-Bench | `scripts/evals/cvbench/` | `scripts/evals/cvbench_api/` |

---

## All-in-one (Benchmark별 전체 실행)

```bash
# 3DSRBench GPU
python scripts/evals/3dsrbench/run_all_models_full.py

# 3DSRBench API
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset

# CV-Bench GPU
python scripts/evals/cvbench/run_all_models_full.py
python scripts/evals/cvbench/run_all_models_full.py --without_prompt

# CV-Bench API
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset
```

---

## Per-model

### 3DSRBench

```bash
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
# Sa2VA, LLaVA4D 동일
```

### CV-Bench

```bash
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt
# Sa2VA, LLaVA4D 동일
```

---

## Benchmarks

| Benchmark | Split | Format | Samples |
|-----------|-------|--------|---------|
| 3DSRBench | test | Multiple choice (A/B/C/D) | ~5.1k |
| CV-Bench | test | Multiple choice | ~2.6k |

See [DATASETS.md](DATASETS.md) for dataset characteristics and selection rationale.

---

## Output structure

```
results/runs/
├── 3dsrbench/
│   ├── qwen3_4b/
│   ├── sa2va/
│   ├── llava4d/
│   └── api_models/
└── cvbench/
    ├── qwen3_4b/
    ├── sa2va/
    ├── llava4d/
    └── api_models/
```
