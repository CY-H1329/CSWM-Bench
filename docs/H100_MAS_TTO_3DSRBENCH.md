# MAS + TTO sur H100 — 3DSRBench

Multi-agent system avec Trust Score (TTO) update, exécutable sur serveur H100.

## Spécialistes

- **llava4d** — LLaVA4D
- **qwen3_4b** — Qwen3-VL-4B
- **sa2va** — Sa2VA
- **spatial_rgpt** — SpatialRGPT (ou spaceOm)

En cas de conflit d'environnement (Sa2VA/SpatialRGPT), utiliser 3 agents :
`--specialist_whitelist qwen3_4b,llava4d,spatial_reasoner`

## Dataset

Utilise `data/dataset/3dsrbench_train_300` (échantillons pré-sélectionnés depuis GitHub).

## Setup (H100)

```bash
# 1. Clone ou pull
cd /path/to/workdir
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS

# 2. Setup (vérifie trust_score, data/dataset)
bash scripts/setup_h100_mas_tto.sh

# 3. Vérifier que data/dataset existe
ls data/dataset/3dsrbench_train_300
# Si absent: python scripts/prepare_train_datasets.py
```

## Exécution

```bash
# Expérience complète (train 50% + test 50%)
bash scripts/run_h100_mas_tto_3dsrbench.sh

# Test rapide (20 échantillons)
bash scripts/run_h100_mas_tto_3dsrbench.sh --test_only --max_samples 20

# Limiter à 100 échantillons
bash scripts/run_h100_mas_tto_3dsrbench.sh --max_samples 100
```

## Options CLI

| Option | Description |
|--------|-------------|
| `--dataset_subdir 3dsrbench_train_300` | Charge depuis data/dataset/ |
| `--use_tto` | Active Trust Score (run_step4 Beta+EMA) |
| `--trust_step 4` | Étape TTO (1–4) |
| `--specialist_whitelist` | Sous-ensemble d'agents |
| `--specialist_offload_after_use` | Économise la mémoire GPU |

## Résultats

Sauvegardés dans `results/mas_v2_tto_3dsrbench/`.
