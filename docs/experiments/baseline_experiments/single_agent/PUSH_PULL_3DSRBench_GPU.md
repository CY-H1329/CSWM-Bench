# Push / Pull — 3DSRBench GPU (Qwen3, LLaVA4D, Sa2VA)

Dataset complet, avec/sans prompt. Chaque commande dans un terminal séparé (GPU parallèle).

---

## 1. Push (Local → GitHub)

```bash
cd ~/Desktop/Spatial_MAS

git add .
git status
git commit -m "3DSRBench GPU: full_dataset, with/without_prompt, 6 runs"
git push origin main
```

---

## 2. Pull (GPU Server ← GitHub)

```bash
cd /path/to/Spatial_MAS   # ou ~/CY/Spatial_MAS

git pull origin main
conda activate spatial_mas
pip install -r requirements.txt

# Datasets (une fois)
python scripts/setup_datasets.py
```

---

## 3. GPU — 6 terminaux (dataset complet, parallèle)

Chaque commande dans un terminal distinct. Sortie : `results/runs/3dsrbench/{model}/full_dataset_{with_prompt|without_prompt}/`

### Terminal 1 — Qwen3 with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
```

### Terminal 2 — Qwen3 without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
```

### Terminal 3 — LLaVA4D with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset
```

### Terminal 4 — LLaVA4D without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset --without_prompt
```

### Terminal 5 — Sa2VA with_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset
```

### Terminal 6 — Sa2VA without_prompt
```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset --without_prompt
```

---

## Structure des sorties

```
results/runs/3dsrbench/
├── qwen3_4b/
│   ├── full_dataset_with_prompt/
│   └── full_dataset_without_prompt/
├── llava4d/
│   ├── full_dataset_with_prompt/
│   └── full_dataset_without_prompt/
└── sa2va/
    ├── full_dataset_with_prompt/
    └── full_dataset_without_prompt/
```
