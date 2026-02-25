# MAS v2 Baseline Experiments (No Trust Score)

Main architecture: Head Agent → 3 Specialists → Final Reasoning Agent.
Benchmarks: 3DSRBench, CV-Bench.
Sample sizes: 10, 50, 100.

## Structure

```
experiments/mas_v2_baseline/
├── README.md
├── mas_v2_baseline.ipynb          # Full experiments (Untitled style, 10/50/100 × 2 benchmarks)
├── mas_v2_quick_test.ipynb       # Quick test (run_mas_test, no train)
├── cvbench_experiments.ipynb     # CV-Bench only
├── 3dsrbench_experiments.ipynb   # 3DSRBench only
├── run_all.py                    # Command: python run_all.py
└── run_h100.sh                   # Command: bash run_h100.sh
```

## Results

Saved to `results/mas_v2_baseline/{benchmark}/{n}samples/{timestamp}/`:
- `summary.json` — accuracy, per-category
- `score_map_final.json` — trained score map
- `train_details.jsonl`, `test_details.jsonl`

## H100 실행

```bash
cd ~/CY/Spatial_MAS
git pull origin main

# Command 1: Full experiments (train+test, 10/50/100 × 2 benchmarks)
bash experiments/mas_v2_baseline/run_h100.sh
# 또는
python experiments/mas_v2_baseline/run_all.py

# Command 2: Quick test (no train, random agents)
python test_final_reasoning_mas_v2.py --benchmark cvbench --max_samples 100 --use_local_reasoning

# Notebook (Untitled.ipynb 스타일)
jupyter notebook experiments/mas_v2_baseline/
# mas_v2_baseline.ipynb — full experiments
# mas_v2_quick_test.ipynb — quick test
```

## Requirements

- `use_local_reasoning=True` — DeepSeek-R1-Distill-Qwen-7B 로컬 (API 불필요)
- `SPATIALRGPT_PATH` — spatial_rgpt specialist 사용 시
- GPU ~24GB+ VRAM 권장
