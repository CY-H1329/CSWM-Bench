# CV-Bench — Résultats par catégorie

Script pour organiser les résultats CV-Bench par **category** et **with_prompt / without_prompt**.

## Structure attendue

```
results/runs/cvbench/
├── llava4d/
│   ├── full_dataset_with_prompt/details.jsonl
│   └── full_dataset_without_prompt/details.jsonl
├── qwen3_4b/
│   ├── full_dataset_with_prompt/
│   └── full_dataset_without_prompt/
├── sa2va/
│   └── ...
└── api_models/
    └── <timestamp>/
        ├── claude_sonnet_4_5_with_prompt/
        ├── claude_sonnet_4_5_without_prompt/
        ├── gpt4o_with_prompt/
        └── ...
```

## Usage

```bash
# Depuis la racine du projet
python scripts/evals/cvbench/aggregate_category_results.py --dir results/runs/cvbench

# Sortie personnalisée
python scripts/evals/cvbench/aggregate_category_results.py --dir results/runs/cvbench --output results/analysis/cvbench_by_category.csv
```

## Output

- **cvbench_by_category.csv** : model, prompt_variant, category, accuracy, correct, total
- **cvbench_by_category.json** : structure complète par modèle et catégorie

## Catégories CV-Bench

Count, Relation, Depth, Distance

## Tableau résumé

Le script affiche :
1. Par catégorie : accuracy (n) pour chaque modèle × with_prompt | without_prompt
2. Overall : accuracy globale par modèle × variant
