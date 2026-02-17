# GQA Evaluation

GQA (lmms-lab/GQA, val_balanced) — Qwen3-VL-4B, Sa2VA-4B, LLaVA4D.

## Categories

- **Semantic**: relation, attribute, object, global, other
- **Structural**: query, verify, choose, logical, compare

Explore categories first:
```bash
python scripts/evals/gqa/explore_categories.py --max_samples 5000
```

---

## Two terminals: with_prompt vs without_prompt

Run in **parallel** (two terminals) to speed up.

### Terminal 1 — WITH prompt (spatial reasoning prompt)

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_mas

# Qwen3-4B
python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset

# Sa2VA-4B (separate run)
python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset

# LLaVA4D (separate run)
python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset
```

### Terminal 2 — WITHOUT prompt (question only)

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_mas

# Qwen3-4B
python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset --without_prompt

# Sa2VA-4B (separate run)
python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset --without_prompt

# LLaVA4D (separate run)
python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset --without_prompt
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

---

## Recommended: run one model per terminal

To avoid GPU memory issues, run each model separately:

| Terminal | Model | Command |
|----------|-------|---------|
| 1 | Qwen3 with_prompt | `python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset` |
| 2 | Qwen3 without_prompt | `python scripts/evals/gqa/run_eval_gqa_qwen3.py --full_dataset --without_prompt` |
| 3 | Sa2VA with_prompt | `python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset` |
| 4 | Sa2VA without_prompt | `python scripts/evals/gqa/run_eval_gqa_sa2va.py --full_dataset --without_prompt` |
| 5 | LLaVA4D with_prompt | `python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset` |
| 6 | LLaVA4D without_prompt | `python scripts/evals/gqa/run_eval_gqa_llava4d.py --full_dataset --without_prompt` |
