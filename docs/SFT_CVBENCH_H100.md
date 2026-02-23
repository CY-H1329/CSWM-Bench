# SFT CV-Bench — H100 Workflow

Pipeline for supervised fine-tuning and evaluation on CV-Bench.

## On H100

```bash
cd ~/CY/Spatial_MAS  # or your clone path
git pull origin main

# 1. Setup (if not done)
conda activate spatial_mas
python scripts/setup_datasets.py --benchmarks cvbench

# 2. Sample dataset
python scripts/sft_cvbench/01_sample_dataset.py

# 3. Train each model × shots
for model in qwen3_4b llava4d sa2va spatialrgpt spatialreasoner; do
  for shots in 10 30 100; do
    python scripts/sft_cvbench/02_train.py --model $model --shots $shots
  done
done

# 4. Evaluate
for model in qwen3_4b llava4d sa2va spatialrgpt spatialreasoner; do
  for shots in 10 30 100; do
    python scripts/sft_cvbench/03_evaluate.py --model $model --shots $shots \
      --checkpoint results/sft_cvbench/checkpoints/${model}_cvbench_${shots}shot
  done
done

# 5. Aggregate
python scripts/sft_cvbench/04_aggregate_results.py
# Output: results/sft_cvbench/results_cvbench_scaling.csv
```

## Push results

```bash
git add results/sft_cvbench/
git add data/sft_cvbench/splits/
git commit -m "SFT CV-Bench: results and splits"
git push origin main
```
