# Spatial_MAS

Spatial reasoning evaluation for vision-language models on **3DSRBench**, **CV-Bench**, and **OMNI3D-BENCH**.

## Benchmarks

| Benchmark | Description | Link |
|-----------|-------------|------|
| **3DSRBench** | 3D spatial reasoning (12 categories) | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) |
| **CV-Bench** | Computer vision reasoning | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| **OMNI3D-BENCH** | 3D understanding | [dmarsili/Omni3D-Bench](https://huggingface.co/datasets/dmarsili/Omni3D-Bench) |

## Models

| Type | Models |
|------|--------|
| **GPU** | Qwen3-VL-4B, Sa2VA-4B, LLaVA4D |
| **API** | Claude Sonnet 4.5, GPT-4o, Gemini Robotics-ER |

---

## Quick start

### 1. Environment

```bash
cd Spatial_MAS
conda env create -f environment.yml
conda activate spatial_mas
python scripts/setup_datasets.py
```

### 2. Run evaluation

**3DSRBench (GPU)**:
```bash
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
```

**3DSRBench (API)**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
```

**MAS pipeline**:
```bash
python run_eval_mas.py --benchmark 3dsrbench --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Environment setup, installation, API keys |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Methodology, benchmarks, evaluation protocol |
| [docs/EXPERIMENT_SETUP.md](docs/EXPERIMENT_SETUP.md) | Experiment structure, reproducibility |
| [docs/DATASETS.md](docs/DATASETS.md) | Per-dataset details and usage |
| [docs/BASELINE_EXPERIMENTS.md](docs/BASELINE_EXPERIMENTS.md) | Baseline experiments, commands, per-model summary |
| [docs/RESULTS_STRUCTURE.md](docs/RESULTS_STRUCTURE.md) | Results folder structure, gather script |
| [docs/H100_PUSH_RESULTS.md](docs/H100_PUSH_RESULTS.md) | H100: push results to GitHub |

---

## Project structure

```
Spatial_MAS/
├── config.yaml              # Config (benchmark, models, eval)
├── run_eval_mas.py          # MAS pipeline
├── run_eval_mas_full.py     # MAS full (all model combos)
├── run_eval_single_3dsrbench.py  # 3DSRBench GPU (all models)
├── scripts/
│   ├── evals/
│   │   ├── 3dsrbench/       # 3DSRBench GPU (per-model)
│   │   └── 3dsrbench_api/   # 3DSRBench API (Claude, GPT-4o, Gemini)
│   └── setup_datasets.py
├── src/
│   ├── benchmarks/         # Dataset loaders
│   ├── models/             # Model runners
│   └── agents/             # MAS prompts
└── docs/
```

---

## GitHub & remote server

- **Push**: `git push origin main`
- **Pull on server**: `git pull origin main`
- See [docs/PUSH_PULL_WORKFLOW.md](docs/PUSH_PULL_WORKFLOW.md) for sync workflow.
- See [docs/GITHUB_AND_H100.md](docs/GITHUB_AND_H100.md) for H100 setup.

---

## Legacy scripts (deprecated)

The following scripts use the deprecated STVQA-7K format and are kept for backward compatibility:
- `run_eval.py`, `run_eval_multiagent.py`, `run_eval_unified.py`, `run_eval_collab.py`
- `analyze_failures.py`, `export_failed_samples.py`

Use the 3DSRBench scripts and MAS pipeline instead.

---

## License

See repository for license information.
