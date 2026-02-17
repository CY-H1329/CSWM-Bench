# Experiment Setup

This document describes the experiment setup for paper submission and reproducibility.

## Directory structure

```
Spatial_MAS/
├── config.yaml              # Main config (benchmark, models, eval params)
├── run_eval_mas.py          # MAS pipeline (1 combination)
├── run_eval_mas_full.py     # MAS full (all model combos)
├── run_eval_single_3dsrbench.py  # 3DSRBench GPU (Qwen3, Sa2VA, LLaVA4D)
├── scripts/
│   ├── evals/
│   │   ├── 3dsrbench/       # 3DSRBench GPU scripts (per-model)
│   │   └── 3dsrbench_api/   # 3DSRBench API (Claude, GPT-4o, Gemini)
│   ├── setup_datasets.py    # Download benchmarks
│   └── run_h100_*.sh        # H100 execution scripts
├── src/
│   ├── benchmarks/         # Dataset loaders (3DSRBench, CV-Bench, GQA)
│   ├── models/             # Model runners (Qwen, LLaVA, Sa2VA, GPT, Gemini)
│   └── agents/             # MAS prompts and pipeline
└── results/                # Run outputs (timestamped)
```

## Benchmark-specific scripts

| Benchmark | Script | Models |
|-----------|--------|--------|
| **3DSRBench (GPU)** | `scripts/evals/3dsrbench/run_eval_3dsrbench_*.py` | Qwen3, Sa2VA, LLaVA4D |
| **3DSRBench (API)** | `scripts/evals/3dsrbench_api/run_eval_api.py` | Claude, GPT-4o, Gemini |
| **All (MAS)** | `run_eval_mas.py`, `run_eval_mas_full.py` | Qwen3, Sa2VA, LLaVA4D |

## Reproducibility

- **Seed**: 42 (default) for dataset sampling.
- **Temperature**: 0.0 for single-agent, 0.4 for multi-agent.
- **Max tokens**: 512 (configurable in `config.yaml`).

## Future experiments

To add a new experiment:

1. Create a script in `scripts/evals/<benchmark>/` or extend existing.
2. Document in `docs/BASELINE_EXPERIMENTS.md`.
3. Add run config to `scripts/evals/<benchmark>/runs_*.json` if needed.
