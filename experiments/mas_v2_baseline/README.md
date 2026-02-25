# MAS v2 Baseline Experiments (No Trust Score)

**Testing only** — no train/test split, no ScoreMap update.

Pipeline: Head Agent → ScoreMap (random) → 3 Specialists → SharedMemory → Final Reasoning Agent

Benchmarks: 3DSRBench, CV-Bench | Sample sizes: 10, 50, 100

## Structure

```
experiments/mas_v2_baseline/
├── README.md
├── cvbench_experiments.ipynb      # CV-Bench: 10, 50, 100 samples
├── 3dsrbench_experiments.ipynb    # 3DSRBench: 10, 50, 100 samples
├── run_all.py                     # Python script (all experiments)
└── run_h100.sh                    # H100 batch script
```

## Results (testing only)

Saved to `results/mas_v2_baseline/{benchmark}/{n}samples/{timestamp}/`:
- `summary.json` — accuracy, per-category
- `details.jsonl` — per-sample results

## H100 실행

```bash
cd ~/CY/Spatial_MAS   # 또는 프로젝트 경로
git pull origin main

# 방법 1: Shell script (6개 실험 순차 실행)
bash experiments/mas_v2_baseline/run_h100.sh

# 방법 2: Python script
python experiments/mas_v2_baseline/run_all.py

# 방법 3: Jupyter notebook
jupyter notebook experiments/mas_v2_baseline/
# cvbench_experiments.ipynb, 3dsrbench_experiments.ipynb 실행
```

## Requirements

- `use_local_reasoning=True` — DeepSeek-R1-Distill-Qwen-7B 로컬 (API 불필요)
- `SPATIALRGPT_PATH` — spatial_rgpt specialist 사용 시
- GPU ~24GB+ VRAM 권장
