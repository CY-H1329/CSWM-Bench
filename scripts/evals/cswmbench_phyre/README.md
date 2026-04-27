## CSWM-Bench (PHYRE track) — pairs contrastives *existantes*

Objectif: construire un dataset **sans générer d’images** “à la main”, en réutilisant
PHYRE (facebookresearch/phyre), qui fournit:

- une **scène initiale** (image)
- une **action** (placement d’objets rouges)
- un **outcome** (succès/échec) via simulation

Nous extrayons des **paires d’actions très proches** \((a_1, a_2)\) mais avec
un outcome **différent** → c’est exactement le signal “minimal action difference → divergence”.

### Installation (sur H100)

PHYRE est un package séparé (Linux/Mac). Voir le guide officiel:

- [facebookresearch/phyre INSTALLATION](https://github.com/facebookresearch/phyre/blob/master/INSTALLATION.md)

Exemple (conda):

```bash
conda create -n phyre python=3.9 -y
conda activate phyre
pip install phyre pillow numpy tqdm
```

### Générer le dataset (JSONL + PNG)

```bash
cd scripts/evals/cswmbench_phyre
python build_cswm_phyre_pairs.py --eval_setup ball_cross_template --split dev --max_tasks 100 --pairs_per_task 3
python make_dataset_viewer.py
```

Outputs (par défaut):
- `data/cswmbench_phyre/cswm_phyre_pairs.jsonl`
- `data/cswmbench_phyre/images/*.png`
- `reports/cswmbench_phyre_viewer/index.html`

### Format JSONL (résumé)

Chaque ligne:
- `task_id`
- `image` (chemin PNG initial)
- `a1`, `a2` (vecteurs d’action PHYRE)
- `gt`:
  - `divergence`: `"different"` (par construction)
  - `a1_outcome`, `a2_outcome`: `"success"` / `"fail"`
  - `action_distance_l2`

