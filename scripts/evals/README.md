# Scripts d'évaluation par rôle

Les scripts Python sont à la racine du projet. Ce dossier documente leur organisation.

| Rôle | Script | Commande type |
|------|--------|---------------|
| Single-agent | `run_eval.py` | `python run_eval.py --models qwen llava --split val` |
| Multi-agent | `run_eval_multiagent.py` | `python run_eval_multiagent.py --models qwen llava` |
| Unified | `run_eval_unified.py` | `python run_eval_unified.py --models qwen llava` |
| Collab | `run_eval_collab.py` | `python run_eval_collab.py --split train` |
| MAS (1 combo) | `run_eval_mas.py` | `python run_eval_mas.py --benchmark stvqa7k --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b` |
| MAS Full | `run_eval_mas_full.py` | `python run_eval_mas_full.py --benchmark stvqa7k` |
| Single 3DSRBench | `run_eval_single_3dsrbench.py` | `python run_eval_single_3dsrbench.py` |
| 3DSRBench Qwen3 | `3dsrbench/run_eval_3dsrbench_qwen3.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py` |
| 3DSRBench Sa2VA | `3dsrbench/run_eval_3dsrbench_sa2va.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py` |
| 3DSRBench LLaVA4D | `3dsrbench/run_eval_3dsrbench_llava4d.py` | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py` |

**3DSRBench** : Exécution séparée par modèle recommandée. L'agent infère la catégorie (Height, Location, Orientation, Multi-Object).

Voir **[docs/EXECUTION_GUIDE.md](../../docs/EXECUTION_GUIDE.md)** pour les détails, prompts et commandes complètes.
