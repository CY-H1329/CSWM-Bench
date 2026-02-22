# Spatial_MAS

Spatial reasoning evaluation for vision-language models on **3DSRBench** and **CV-Bench**.

## Benchmarks

| Benchmark | Description | Link |
|-----------|-------------|------|
| **3DSRBench** | 3D spatial reasoning (12 categories) | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) |
| **CV-Bench** | Computer vision reasoning | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |

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
python scripts/evals/3dsrbench/run_all_models_full.py
```

**3DSRBench (API)**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
```

**CV-Bench (GPU)**:
```bash
python scripts/evals/cvbench/run_all_models_full.py
```

---

## Documentation

→ **[docs/README.md](docs/README.md)** — Full index

| Document | Description |
|----------|-------------|
| [BASELINE_EXPERIMENTS](docs/BASELINE_EXPERIMENTS.md) | Setup, all commands (per-model, all-in-one, with/without prompt) |
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
├── spatial_aomas/              # Trust Score (Step 1~4, score-based agent selection)
├── results_summary/             # Aggregated results (tracked)
├── scripts/
│   ├── evals/3dsrbench/         # GPU (Qwen3, Sa2VA, LLaVA4D)
│   ├── evals/3dsrbench_api/     # API (Claude, GPT-4o, Gemini)
│   ├── evals/cvbench/           # GPU
│   ├── evals/cvbench_api/       # API
│   ├── gather_results_summary.py
│   └── setup_datasets.py
├── src/benchmarks/              # Dataset loaders
├── src/models/                  # Model runners
└── docs/                        # Documentation (see docs/README.md)
```

---

## GitHub & remote server

- **Push**: `git push origin main`
- **Pull**: `git pull origin main`
- See [docs/GIT_AND_SERVER.md](docs/GIT_AND_SERVER.md) for setup and workflow.

---

## License

See repository for license information.
