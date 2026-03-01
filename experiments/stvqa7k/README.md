# STVQA-7K Evaluation

Évaluation des 5 modèles VLM sur **STVQA-7K** (split `val`, 692 exemples).

- **Dataset**: [hunarbatra/STVQA-7K](https://huggingface.co/datasets/hunarbatra/STVQA-7K/viewer/default/val)
- **Modèles** : Qwen-3.0-VL 4B, LLaVA-4D, SpatialReasoner, SpatialRGPT, Sa2VA

## Workflow H100 (Git push → pull → run)

### 1. Local — Push sur GitHub

```bash
cd Spatial_MAS
git add experiments/stvqa7k/ scripts/stvqa7k/ src/data.py
git commit -m "STVQA-7K eval: 5 models on val"
git push
```

### 2. H100 — One-time setup

```bash
git clone https://github.com/AnjieCheng/SpatialRGPT.git
export SPATIALRGPT_PATH=/path/to/SpatialRGPT
conda activate srgpt   # has s2wrapper; or: pip install git+https://github.com/bfshi/scaling_on_scales
```

### 3. H100 — Pull and run

```bash
cd ~/Spatial_MAS
git pull
export SPATIALRGPT_PATH=/path/to/SpatialRGPT

bash experiments/stvqa7k/run_h100.sh
# Quick test: MAX_SAMPLES=10 bash experiments/stvqa7k/run_h100.sh
```

### 4. Results

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
