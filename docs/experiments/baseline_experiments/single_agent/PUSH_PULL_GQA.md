# Push / Pull — GQA (Qwen3, Sa2VA, LLaVA4D)

Full dataset, with/without prompt. Run each command in a separate terminal.

---

## 1. Push (Local → GitHub)

```bash
cd ~/Desktop/Spatial_MAS

git add .
git status
git commit -m "GQA: full_dataset, with/without_prompt, 6 runs"
git push origin main
```

---

## 2. Pull (GPU Server ← GitHub)

```bash
cd /path/to/Spatial_MAS   # e.g. ~/CY/Spatial_MAS

git pull origin main
conda activate spatial_mas
pip install -r requirements.txt

# Datasets (once)
python scripts/setup_datasets.py
```

---

## 3. Test (30 samples) — verify before full run

Run in **two terminals** (with_prompt vs without_prompt). Each model separately to avoid GPU OOM.

### Terminal 1 — WITH prompt (30 samples)

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

# Qwen3-4B
python scripts/evals/gqa/run_eval_gqa_qwen3.py --max_samples 30

# Sa2VA-4B (after Qwen3 finishes)
python scripts/evals/gqa/run_eval_gqa_sa2va.py --max_samples 30

# LLaVA4D (after Sa2VA finishes)
python scripts/evals/gqa/run_eval_gqa_llava4d.py --max_samples 30
```

### Terminal 2 — WITHOUT prompt (30 samples)

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

# Qwen3-4B
python scripts/evals/gqa/run_eval_gqa_qwen3.py --max_samples 30 --without_prompt

# Sa2VA-4B (after Qwen3 finishes)
python scripts/evals/gqa/run_eval_gqa_sa2va.py --max_samples 30 --without_prompt

# LLaVA4D (after Sa2VA finishes)
python scripts/evals/gqa/run_eval_gqa_llava4d.py --max_samples 30 --without_prompt
```

---

## 4. Full dataset — 6 terminaux (parallèle)

Each command in a **separate terminal**. Output: `results/runs/gqa/{model}/full_dataset_{with_prompt|without_prompt}/`

### Terminal 1 — Qwen3 with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset
```

### Terminal 2 — Qwen3 without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset --without_prompt
```

### Terminal 3 — Sa2VA with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset
```

### Terminal 4 — Sa2VA without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset --without_prompt
```

### Terminal 5 — LLaVA4D with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset
```

### Terminal 6 — LLaVA4D without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset --without_prompt
```

---

## 5. Gather results & push

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/gather_results_summary.py

git add results_summary/
git status
git commit -m "GQA results: Qwen3, Sa2VA, LLaVA4D"
git push origin main
```

---

## Output structure

```
results/runs/gqa/
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
