# CV-Bench Evaluation

CV-Bench (nyu-visionx/CV-Bench) — Qwen3-VL-4B, Sa2VA-4B, LLaVA4D.

## Task types

- **2D**: spatial relationships, object counting (ADE20K, COCO)
- **3D**: depth order, relative distance (Omni3D)

Explore actual task names from the dataset:
```bash
python scripts/evals/cvbench/explore_categories.py --max_samples 500
```

---

## Push / Pull

See **docs/experiments/baseline_experiments/single_agent/PUSH_PULL_CVBench.md**.

---

## Test (30 samples) — verify before full run

### Terminal 1 — WITH prompt
```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --max_samples 30
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --max_samples 30
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --max_samples 30
```

### Terminal 2 — WITHOUT prompt
```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --max_samples 30 --without_prompt
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --max_samples 30 --without_prompt
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --max_samples 30 --without_prompt
```

---

## Full dataset — Two terminals

### Terminal 1 — WITH prompt
```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset
```

### Terminal 2 — WITHOUT prompt
```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset --without_prompt
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset --without_prompt
```

---

## Output structure

```
results/runs/cvbench/
├── qwen3_4b/
│   ├── full_dataset_with_prompt/
│   └── full_dataset_without_prompt/
├── sa2va/
│   ├── full_dataset_with_prompt/
│   └── full_dataset_without_prompt/
└── llava4d/
    ├── full_dataset_with_prompt/
    └── full_dataset_without_prompt/
```
