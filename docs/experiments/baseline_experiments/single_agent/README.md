# Spatial_MAS – Expériences et tests

Ce dossier documente les expériences réalisées sur STVQA-7K (Spatial VQA).

---

## Vue d'ensemble

| Type | Script | Description |
|------|--------|-------------|
| **Single** | `run_eval.py` | Un agent par modèle (Qwen, LLaVA, GPT, Gemini) |
| **Multi-agent** | `run_eval_multiagent.py` | 3 agents identiques, vote majoritaire |
| **Unified** | `run_eval_unified.py` | Single → Multi en une seule exécution, comparaison |
| **Collab** | `run_eval_collab.py` | Qwen + LLaVA : 2 agents, consensus ou tie-break |

---

## Scripts H100

| Script | Commande | Résultat |
|--------|----------|----------|
| Single + analyse | `bash scripts/run_h100.sh` | `results/YYYYMMDD_HHMMSS/` |
| Multi-agent | `bash scripts/run_h100_multiagent.sh` | `results/YYYYMMDD_HHMMSS_unified/` |
| Collab | `bash scripts/run_h100_collab.sh` | `results/YYYYMMDD_HHMMSS_collab/` |

---

## Paramètres clés (config.yaml)

- `eval.temperature` : 0 (single, déterministe)
- `eval.multi_agent_temperature` : 0.4 (multi-agent, diversité)
- `eval.collab_temperature` : 0.4 (collab discuss)
- `eval.top_k`, `eval.top_p` : 50, 0.9 (sampling)
- `dataset.max_per_category` : 7 (test rapide) ou 100 (exp complet)

---

## Modèles

- **Qwen** : `Qwen/Qwen2.5-VL-7B-Instruct`
- **LLaVA** : `llava-hf/llava-v1.6-mistral-7b-hf` (NeXT) ou `llava-hf/llava-1.5-7b-hf`

---

## Fichiers de sortie

- `*_preds.jsonl` : prédictions par échantillon
- `*_results.json` : précision, by_category
- `conversations/` : logs multi-agent (question, 3 réponses, majorité)
- `wrong_comparison.txt` : single vs multi (both_wrong, recovered, regressed)
- `comparison_collab.txt` : qwen_only vs llava_only vs qwen_llava_collab

---

## MAS Prompts

Prompts détaillés (markdown, structurés) : **[MAS_PROMPTS.md](./MAS_PROMPTS.md)**  
Fichier source utilisé par le code : `src/agents/prompts.yaml`

---

## Step 1 – Architecture Head → Perception → Reasoning

Voir **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** pour :

- Chaîne : Head-Agent → Perception Agent → Reasoning Agent
- 4 benchmarks : OMNI3D-BENCH, CV-Bench, 3DSRBench, STVQA-7K
- Phase 1 : Qwen-3.0 4B, Llava-4D, Sa2VA (agents identiques + 27 combinaisons)
- Phase 2 : Claude, GPT-4o, Deepseek, Gemini

Organisation des résultats : `runs/<benchmark>/<head>_<perception>_<reasoning>/`

---

## Nouvelles expériences

Pour ajouter une nouvelle expérience :

1. Créer un script dans `docs/experiments/` ou à la racine
2. Documenter ici (nom, commande, paramètres)
3. Optionnel : ajouter un script dans `scripts/` pour H100
