# Agrégation par catégorie — 3DSRBench

Script : `aggregate_category_performance.py`

Calcule l'accuracy par catégorie (12 catégories 3DSRBench) pour chaque run, à partir des `details.jsonl`.

## Chemins requis

Exécuter depuis la racine du projet (`~/CY/Spatial_MAS` ou `~/Desktop/Spatial_MAS`).

| Type | --dir | Contenu attendu |
|------|-------|-----------------|
| **API** | `results/runs/3dsrbench/api_models/20260217_023920` | Sous-dirs : `claude_sonnet_4_5_with_prompt`, `gpt4o_without_prompt`, etc. (chacun avec `details.jsonl`) |
| **API** | `results/runs/3dsrbench/api_models/full_dataset` | Idem |
| **GPU** | `results/runs/3dsrbench` | Sous-dirs : `qwen3_4b/`, `llava4d/`, `sa2va/` avec `full_dataset_with_prompt/`, `full_dataset_without_prompt/` |

## Usage

```bash
cd ~/CY/Spatial_MAS   # ou chemin du projet

# API — run spécifique (timestamp)
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench/api_models/20260217_023920 --mode api

# API — full_dataset
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench/api_models/full_dataset --mode api

# GPU
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --mode gpu

# GPU — 6 runs exacts (éditer runs_gpu_6.json pour vos chemins)
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --runs_file scripts/evals/3dsrbench/runs_gpu_6.json

# API — par modèle (quand Claude pas encore fini)
API_DIR="results/runs/3dsrbench/api_models/20260216_121420"
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir $API_DIR --runs_file scripts/evals/3dsrbench/runs_api_gpt4o.json --output $API_DIR/category_gpt4o.csv
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir $API_DIR --runs_file scripts/evals/3dsrbench/runs_api_gemini.json --output $API_DIR/category_gemini.csv
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir $API_DIR --runs_file scripts/evals/3dsrbench/runs_api_claude.json --output $API_DIR/category_claude.csv

# API — tous (quand Claude fini)
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir $API_DIR --runs_file scripts/evals/3dsrbench/runs_api_all.json

# Auto (recherche récursive de tous les details.jsonl)
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --mode auto

# Sortie personnalisée
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir ... --output category_results.csv --output_dir results/analysis
```

## Sorties

- `category_performance.csv` : model, category, accuracy, correct, total
- `category_performance.json` : structure complète par modèle et catégorie
