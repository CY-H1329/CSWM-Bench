# Baseline Evaluation Scripts

3DSRBench and CV-Bench single-agent baseline. See [docs/BASELINE_EXPERIMENTS.md](../../docs/BASELINE_EXPERIMENTS.md).

## All-in-one (Benchmark별 전체 실행)

| Benchmark | Command |
|-----------|---------|
| 3DSRBench GPU | `python scripts/evals/3dsrbench/run_all_models_full.py` |
| 3DSRBench API | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset` |
| CV-Bench GPU | `python scripts/evals/cvbench/run_all_models_full.py` |
| CV-Bench API | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset` |

## Per-model

| Benchmark | GPU | API |
|-----------|-----|-----|
| 3DSRBench | `3dsrbench/run_eval_3dsrbench_qwen3.py`, `run_eval_3dsrbench_sa2va.py`, `run_eval_3dsrbench_llava4d.py` | `3dsrbench_api/run_eval_api.py --model ...` |
| CV-Bench | `cvbench/run_eval_cvbench_qwen3.py`, `run_eval_cvbench_sa2va.py`, `run_eval_cvbench_llava4d.py` | `cvbench_api/run_eval_api.py --model ...` |
