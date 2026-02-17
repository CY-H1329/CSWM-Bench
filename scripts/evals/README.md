# Scripts d'évaluation par rôle

Les scripts Python sont à la racine du projet. Ce dossier documente leur organisation.

| Rôle | Script | Commande type |
|------|--------|---------------|
| Single-agent | `run_eval.py` | `python run_eval.py --models qwen llava --split val` |
| Multi-agent | `run_eval_multiagent.py` | `python run_eval_multiagent.py --models qwen llava` |
| Unified | `run_eval_unified.py` | `python run_eval_unified.py --models qwen llava` |
| Collab | `run_eval_collab.py` | `python run_eval_collab.py --split train` |
| MAS (1 combo) | `run_eval_mas.py` | `python run_eval_mas.py --benchmark 3dsrbench --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b` |
| MAS Full | `run_eval_mas_full.py` | `python run_eval_mas_full.py --benchmark 3dsrbench` |
| **CV-Bench** | `scripts/evals/cvbench/` | Qwen3, Sa2VA, LLaVA4D — full_dataset, with/without prompt |
| Single 3DSRBench | `run_eval_single_3dsrbench.py` | `python run_eval_single_3dsrbench.py` |
| 3DSRBench Qwen3 | `3dsrbench/run_eval_3dsrbench_qwen3.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py` |
| 3DSRBench Sa2VA | `3dsrbench/run_eval_3dsrbench_sa2va.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py` |
| 3DSRBench LLaVA4D | `3dsrbench/run_eval_3dsrbench_llava4d.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py` |
| 3DSRBench Full (3 modèles) | `3dsrbench/run_all_models_full.py` | `python scripts/evals/3dsrbench/run_all_models_full.py` |
| **3DSRBench API** (100 samples) | `3dsrbench_api/run_eval_api.py` | `python scripts/evals/3dsrbench_api/run_eval_api.py` |
| **CV-Bench** | `cvbench/run_eval_cvbench_*.py` | `python scripts/evals/cvbench/run_eval_cvbench_qwen3.py`, etc. |
| **CV-Bench API** | `cvbench_api/run_eval_api.py` | `python scripts/evals/cvbench_api/run_eval_api.py --test` / `--full_dataset` |

**3DSRBench** : GPU (Qwen3, Sa2VA, LLaVA4D) — exécution séparée recommandée.  
**3DSRBench API** : Claude, GPT-4o, Gemini — 100 samples.  
**CV-Bench** : Qwen3, Sa2VA, LLaVA4D — full_dataset, with_prompt / without_prompt. Voir `cvbench/README.md`.  
**CV-Bench API** : Claude, GPT-4o, Gemini — `--test` (10 samples) / `--full_dataset`. Voir `cvbench_api/README.md`.

Voir **[docs/EXECUTION_GUIDE.md](../../docs/EXECUTION_GUIDE.md)** pour les détails, prompts et commandes complètes.
