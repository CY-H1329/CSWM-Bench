# Push / Pull + commandes API (3DSRBench & CV-Bench)

Dépôt : [https://github.com/CY-H1329/Spatial_MAS](https://github.com/CY-H1329/Spatial_MAS)

Exécuter depuis la **racine du repo** (dossier qui contient `src/`, `scripts/`, `data/`).

---

## 1. Pull (serveur / autre machine ← GitHub)

```bash
cd /chemin/vers/Spatial_MAS
git pull origin main
```

Première fois :

```bash
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS
```

Dépendances minimales pour l’éval API :

```bash
pip install datasets huggingface_hub openai anthropic google-genai pyyaml tqdm pillow requests
# ou : pip install -r requirements.txt   # si vous utilisez le fichier du projet
```

Variables d’environnement (selon les modèles) : `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.  
Optionnel : `HF_TOKEN` ou `HUGGING_FACE_HUB_TOKEN` si le dataset HF le demande.

---

## 2. Push (machine locale → GitHub)

```bash
cd /chemin/vers/Spatial_MAS
git status
git add scripts/evals/3dsrbench_api/ src/benchmarks/ scripts/evals/3dsrbench/common.py src/data.py
# ou : git add -A
git commit -m "API eval: 3DSRBench/CV-Bench, per-category metrics"
git push origin main
```

(Branche par défaut `main` ; adaptez si vous utilisez `master`.)

---

## 3. 3DSRBench seul (API)

`--benchmark` par défaut = `3dsrbench`. Sorties : `results/runs/3dsrbench/api_models/...`

```bash
# Sous-échantillon (ex. 1000 tirages HF ou frozen 500 selon config)
python scripts/evals/3dsrbench_api/run_eval_api.py

# Dataset complet HF test (pas le frozen 500 local)
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset

# Un seul modèle + prompt spatial uniquement
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark 3dsrbench --full_dataset --model gpt_5_2 --prompt_variant with_prompt

# Script H100 / Linux (GPT-5.2, full 3DSRBench)
bash scripts/evals/3dsrbench_api/h100_run_gpt52_3dsrbench.sh
```

---

## 4. CV-Bench seul (API)

Toujours passer `--benchmark cvbench`. Sorties : `results/runs/cvbench/api_models/...`

```bash
# Défaut : frozen local `cvbench_400` si présent, sinon HF
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench

# Test HF complet (~2638)
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench --full_dataset

# GPT-5.2 uniquement, full CV-Bench
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench --full_dataset --model gpt_5_2 --prompt_variant with_prompt

bash scripts/evals/3dsrbench_api/h100_run_gpt52_cvbench.sh
```

---

## 4b. MMSI-Bench seul (API, multi-images)

Dataset Hugging Face : `RunsenXu/MMSI-Bench` (1000 questions, plusieurs images par item). Pas de frozen local dans ce dépôt. Sorties : `results/runs/mmsibench/api_models/...`

```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark mmsibench
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark mmsibench --full_dataset --model gpt_5_2 --prompt_variant with_prompt
bash scripts/evals/3dsrbench_api/h100_run_gpt52_mmsibench.sh
```

Les modèles **Claude / GPT / Gemini / OpenRouter** envoient toutes les images dans un même message. **DeepSeek** (`api.deepseek.com` /v1/vision) : les images sont **concaténées verticalement** en une seule si plusieurs cadres.

---

## 5. Résultats **par catégorie** (tâche)

Pour **chaque** benchmark, le script affiche dans le terminal des lignes du type :

`Per-category (answer acc):` puis `nom_catégorie: 0.xxxx (correct/total)`.

Fichiers :

| Fichier | Contenu |
|--------|---------|
| `.../<run_key>/results.json` | `per_category_answer_accuracy` : par clé catégorie (`accuracy`, `correct`, `n`) + `category_cls_accuracy` |
| `.../summary.txt` | Tableau global + section **Per-category answer accuracy** par modèle |

Exemple (chemin typique après un run) :

```bash
RUN=results/runs/3dsrbench/api_models/full_dataset/gpt_5_2_with_prompt
cat "$RUN/results.json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('per_category_answer_accuracy',{}), indent=2))"
```

CV-Bench : catégories **Count**, **Relation**, **Depth**, **Distance** (colonne `task` du dataset).  
3DSRBench : catégories fines du benchmark (ex. `location_above`, …).  
MMSI-Bench : 11 valeurs `question_type` du HF (ex. `Motion (Cam.)`, `MSR`, `Positional Relationship (Cam.–Cam.)`, …).

---

## 6. Ne pas mélanger les deux benchmarks dans un seul run

Un lancement = un `--benchmark`. Pour enchaîner les deux :

```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark 3dsrbench --full_dataset --model gpt_5_2 --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench     --full_dataset --model gpt_5_2 --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark mmsibench --full_dataset --model gpt_5_2 --prompt_variant with_prompt
```
