# STVQA-7K Evaluation

Évaluation des 5 modèles VLM sur **STVQA-7K** (split `val`, 692 exemples).

- **Dataset** : [hunarbatra/STVQA-7K](https://huggingface.co/datasets/hunarbatra/STVQA-7K)
- **Modèles** : Qwen-3.0-VL 4B, LLaVA-4D, SpatialReasoner, SpatialRGPT, Sa2VA

## Workflow H100 (Git push → pull → run)

### 1. Local — Push sur GitHub

```bash
cd Spatial_MAS
git add experiments/stvqa7k/ scripts/stvqa7k/ src/data.py
git commit -m "STVQA-7K eval: 5 models on val"
git push
```

### 2. H100 — Pull et exécution

```bash
cd ~/Spatial_MAS   # ou votre chemin
git pull

# SpatialRGPT (requis pour spatialrgpt). Si Python < 3.10, le script applique un patch automatique.
export SPATIALRGPT_PATH=/path/to/SpatialRGPT

# Test rapide (10 samples)
MAX_SAMPLES=10 bash experiments/stvqa7k/run_h100.sh

# Évaluation complète (692 samples)
bash experiments/stvqa7k/run_h100.sh
```

### 3. Résultats

- `results/stvqa7k/all_results.json` — résumé global
- `results/stvqa7k/<model>/results.json` — par modèle
- `results/stvqa7k/<model>/predictions.jsonl` — prédictions détaillées

## Évaluation d’un seul modèle

```bash
python scripts/stvqa7k/eval_stvqa7k.py --model qwen3_4b --output_dir results/stvqa7k
```

## Variables d’environnement

| Variable | Description |
|----------|-------------|
| `SPATIALRGPT_PATH` | Chemin vers le repo SpatialRGPT (requis pour `spatialrgpt`) |
| `MAX_SAMPLES` | Limite le nombre d’exemples (ex. `10` pour test rapide) |
| `OUTPUT_DIR` | Répertoire de sortie (défaut: `results/stvqa7k`) |
