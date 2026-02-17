# Environment Setup

This document describes how to set up the environment for running Spatial_MAS experiments.

## Requirements

- Python 3.10+
- CUDA (for GPU models)
- Conda (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS
```

### 2. Create Conda environment

```bash
conda env create -f environment.yml
conda activate spatial_mas
```

### 3. Adjust CUDA version (if needed)

For H100 or other CUDA 12.x GPUs, you may need to modify `environment.yml` to use the appropriate `cudatoolkit` version.

### 4. Download datasets

```bash
python scripts/setup_datasets.py
# Or specific benchmarks:
python scripts/setup_datasets.py --benchmarks 3dsrbench cvbench gqa
```

Datasets are cached in `~/.cache/huggingface/datasets/` and reused automatically.

### 5. API keys (for API-based models)

Create a `.env` file in the project root:

```bash
# For GPT-4o
export OPENAI_API_KEY=sk-...

# For Claude
export ANTHROPIC_API_KEY=sk-ant-...

# For Gemini
export GEMINI_API_KEY=...

# For DeepSeek-VL (optional)
export DEEPSEEK_API_KEY=...
```

Or set them in your shell before running.

### 6. Jupyter kernel (optional)

To use the environment in Jupyter:

```bash
python -m ipykernel install --user --name spatial_mas --display-name "spatial_mas"
```

## GPU configuration

- Set `device: "cuda"` in `config.yaml` for GPU inference.
- For multi-GPU systems: `CUDA_VISIBLE_DEVICES=0 python run_eval.py ...`
- Flash Attention 2 and TF32 are enabled by default for H100/Ampere+ GPUs.
