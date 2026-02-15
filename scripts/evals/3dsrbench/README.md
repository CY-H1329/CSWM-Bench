# 3DSRBench — Scripts par modèle

Chaque modèle s'exécute **séparément** pour éviter toute interférence (résultats identiques, fuite de mémoire, etc.).

## Catégories 3DSRBench (officielles)

- **Height** — Position verticale des objets en 3D
- **Location** — Positions et relations spatiales
- **Orientation** — Orientation des objets en 3D
- **Multi-Object** — Raisonnement multi-objets

L'agent **infère lui-même** la catégorie (STEP 1 — TASK CLASSIFICATION). La catégorie n'est pas fournie.

## Commandes

```bash
# Depuis la racine du projet
cd /path/to/Spatial_MAS

# Qwen3-4B uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 50 --seed 42

# Sa2VA uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --max_samples 50 --seed 42

# LLaVA4D uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --max_samples 50 --seed 42
```

## Métriques

- **Answer Accuracy** : précision sur la réponse finale (A/B/C/D)
- **Category Cls Accuracy** : précision de la classification de tâche (STEP 1) vs GT du dataset (Height, Location, Orientation, Multi-Object)

## Sorties

```
results/runs/3dsrbench/
├── qwen3_4b/<timestamp>/
│   ├── responses/sample_*.txt
│   ├── details.jsonl
│   └── results.json
├── sa2va/<timestamp>/
│   └── ...
└── llava4d/<timestamp>/
    └── ...
```

## Prompt utilisé

Voir `common.py` — l'agent classifie la question en STEP 1 (Height, Location, Orientation, Multi-Object) sans recevoir la catégorie.
