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

→ **[docs/README.md](docs/README.md)** — Full index

| Document | Description |
|----------|-------------|
| [ENVIRONMENT](docs/ENVIRONMENT.md) | Setup, installation, API keys |
| [GIT_AND_SERVER](docs/GIT_AND_SERVER.md) | Git workflow, H100 setup |
| [METHODOLOGY](docs/METHODOLOGY.md) | Benchmarks, evaluation protocol |
| [BASELINE_EXPERIMENTS](docs/BASELINE_EXPERIMENTS.md) | Experiments, commands |
| [RESULTS](docs/RESULTS_STRUCTURE.md) | Results structure, gather script |

---

## Project structure

```
Spatial_MAS/
├── config.yaml
├── run_eval_mas.py              # MAS pipeline
├── run_eval_mas_full.py         # MAS full
├── run_eval_single_3dsrbench.py # 3DSRBench GPU
├── results_summary/             # Aggregated results (tracked)
├── scripts/
│   ├── evals/3dsrbench/         # GPU (Qwen3, Sa2VA, LLaVA4D)
│   ├── evals/3dsrbench_api/     # API (Claude, GPT-4o, Gemini)
│   ├── gather_results_summary.py
│   └── setup_datasets.py
├── src/benchmarks/              # Dataset loaders
├── src/models/                  # Model runners
├── src/agents/                  # MAS prompts
└── docs/                        # Documentation (see docs/README.md)
```

---

## GitHub & remote server

- **Push**: `git push origin main`
- **Pull**: `git pull origin main`
- See [docs/GIT_AND_SERVER.md](docs/GIT_AND_SERVER.md) for setup and workflow.

---

## Legacy scripts (deprecated)

The following scripts use the deprecated STVQA-7K format and are kept for backward compatibility:
- `run_eval.py`, `run_eval_multiagent.py`, `run_eval_unified.py`, `run_eval_collab.py`
- `analyze_failures.py`, `export_failed_samples.py`

Use the 3DSRBench scripts and MAS pipeline instead.

---

## License

See repository for license information.
