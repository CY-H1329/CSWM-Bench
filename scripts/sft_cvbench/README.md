# SFT CV-Bench Pipeline

Supervised fine-tuning and evaluation of VLMs on CV-Bench.
Analyzes how performance scales with training size (10/30/100 shots per task).

## Models

- qwen3_4b
- llava4d
- sa2va
- spatialrgpt
- spatialreasoner

## Prerequisites

```bash
pip install datasets pyyaml peft  # peft for LoRA
python scripts/setup_datasets.py --benchmarks cvbench  # cache CV-Bench
```

For qwen3_4b training: `transformers>=4.51` (Qwen3VLForConditionalGeneration)

Uses custom training loop (no Trainer) to avoid sklearn/llava version conflicts.

## Quick Start

```bash
# 1. Sample dataset (stratified, reproducible)
python scripts/sft_cvbench/01_sample_dataset.py

# 2. Train (per model, per shots)
python scripts/sft_cvbench/02_train.py --model qwen3_4b --shots 10

# 3. Evaluate on human_selected_test_set
python scripts/sft_cvbench/03_evaluate.py --model qwen3_4b --shots 10 --checkpoint results/sft_cvbench/checkpoints/qwen3_4b_cvbench_10shot

# 4. Aggregate results
python scripts/sft_cvbench/04_aggregate_results.py
```

## Output

- `data/sft_cvbench/splits/` — train/test indices (JSON)
- `results/sft_cvbench/checkpoints/` — trained checkpoints
- `results/sft_cvbench/results_cvbench_scaling.csv` — Model | Shots | 2D Acc | 3D Acc | Overall Acc

## Splits

| Split | Description |
|-------|-------------|
| human_test | 300 2D + 300 3D, no overlap with training |
| train_10 | 10 per task (40 total) |
| train_30 | 30 per task (120 total) |
| train_100 | 100 per task (400 total) |

## H100

```bash
# Run full pipeline
bash scripts/sft_cvbench/run_h100_sft.sh
```

## Fairness

- Same training samples for all models (fixed seed)
- Same evaluation set
- Identical hyperparameters across models
