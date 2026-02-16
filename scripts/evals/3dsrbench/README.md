# 3DSRBench — Scripts par modèle

Chaque modèle s'exécute **séparément** pour éviter toute interférence (résultats identiques, fuite de mémoire, etc.).

## Catégories 3DSRBench (12, fine-grained)

- location_above, height_higher, location_closer_to_camera
- multi_object_closer_to, orientation_on_the_left, multi_object_facing
- multi_object_same_direction, orientation_in_front_of
- multi_object_viewpoint_towards_object, orientation_viewpoint
- location_next_to, multi_object_parallel

L'agent **infère lui-même** la catégorie (STEP 1 — TASK CLASSIFICATION). La catégorie n'est pas fournie.

## Commandes

### Dataset complet (3 modèles)

```bash
# Exécute Qwen3, Sa2VA, LLaVA4D sur le dataset complet (sans --max_samples)
python scripts/evals/3dsrbench/run_all_models_full.py
python scripts/evals/3dsrbench/run_all_models_full.py --seed 42

# Dataset complet + with/without prompt (6 runs, terminaux séparés)
# Voir docs/experiments/baseline_experiments/single_agent/PUSH_PULL_3DSRBench_GPU.md
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
```

### Par modèle (test rapide ou ciblé)

```bash
# Qwen3-4B uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py              # dataset complet
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 50  # test rapide

# Sa2VA uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --max_samples 50

# LLaVA4D uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --max_samples 50
```

## Métriques

- **Answer Accuracy** : précision sur la réponse finale (A/B/C/D)
- **Category Cls Accuracy** : précision de la classification de tâche (STEP 1) vs GT du dataset (12 catégories fine-grained)

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
