# SFT CV-Bench Pipeline

Supervised fine-tuning and evaluation of VLMs on CV-Bench.
Analyzes how performance scales with training size (10/30/100 shots per task).

## Models

| Model | Status | Notes |
|-------|--------|-------|
| qwen3_4b | ✅ | Qwen3-VL-4B, LoRA |
| llava4d | ✅ | LLaVA-1.5 (llava-1.5-7b-hf) — LLaVA-NeXT has token/feature mismatch |
| sa2va | ✅ | Sa2VA-4B, LoRA on LLM |
| spatialreasoner | ✅ | SpatialReasoner (Qwen2.5-VL), LoRA |
| spatialrgpt | ⚠️ | Requires `SPATIALRGPT_PATH`; use official repo for full training |

## Prerequisites

```bash
pip install datasets pyyaml peft  # peft for LoRA
python scripts/setup_datasets.py --benchmarks cvbench  # cache CV-Bench
```

- **qwen3_4b**: `transformers>=4.57` (Qwen3VLForConditionalGeneration)
- **llava4d**: Uses LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`); LLaVA-NeXT has a known token/feature mismatch
- **sa2va**: `trust_remote_code=True` (custom modeling)
- **spatialrgpt**: `export SPATIALRGPT_PATH=/path/to/SpatialRGPT` (clone [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT))

Uses custom training loop (no Trainer) to avoid sklearn/llava version conflicts.

## Quick Start

```bash
# 프로젝트 루트에서 실행
cd Spatial_MAS  # 또는 ~/CY/Spatial_MAS

# 0. (선택) 모델 로드/추론 확인
python scripts/sft_cvbench/00_verify_models.py

# 1. 학습만 실행 (sample + train 4 models)
bash scripts/sft_cvbench/run_train_all.sh 10

# 2. 또는 전체 파이프라인 (train + eval + aggregate)
bash scripts/sft_cvbench/run_full_pipeline.sh 10

# 개별 학습
python scripts/sft_cvbench/01_sample_dataset.py
python scripts/sft_cvbench/02_train.py --model qwen3_4b --shots 10
python scripts/sft_cvbench/02_train.py --model llava4d --shots 10
python scripts/sft_cvbench/02_train.py --model sa2va --shots 10
python scripts/sft_cvbench/02_train.py --model spatialreasoner --shots 10
```

## Git Push / Pull

```bash
# Pull (최신 코드 가져오기)
cd Spatial_MAS && git pull origin main

# Push (수정 후 업로드)
git add .
git commit -m "message"
git push origin main
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
