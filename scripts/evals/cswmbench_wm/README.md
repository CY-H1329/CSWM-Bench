## CSWM-Bench (World-Model track)

Cette variante sert à évaluer des **world models** (ex: Dreamer) sur des tâches
où un **delta d'action minimal** change (ou non) le futur.

### Principe

- On génère des scènes **paramétriques** avec une dynamique simple (2D, déterministe).
- On considère une **paire d'actions** \(a_1, a_2\) (ex: pousser 2cm vs 6cm).
- Un world model doit prédire:
  - si les futurs **divergent** (étiquette `same|different`)
  - et/ou les **événements** (`hit|clear`, `both_safe|both_fall|push2_safe_push6_fall`)

### Scripts

- `generate_cswm_wm.py`: génère `data/cswmbench_wm/cswmbench_wm.jsonl`
- `run_eval_cswm_wm.py`: évalue un "predictor" (oracle, random, ou adaptateur Dreamer)

### Lancer

```bash
python scripts/evals/cswmbench_wm/generate_cswm_wm.py
python scripts/evals/cswmbench_wm/run_eval_cswm_wm.py --predictor oracle
python scripts/evals/cswmbench_wm/run_eval_cswm_wm.py --predictor random
```

### Brancher Dreamer (adaptateur)

Le script expose une interface minimale: `predict_rollout(state, action, horizon)`.
Tu peux implémenter `DreamerPredictor` dans `predictors.py` en chargeant ton checkpoint.

