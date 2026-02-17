# Commands Reference

All evaluation commands for 3DSRBench and CV-Bench.  
Run from project root: `cd ~/CY/Spatial_MAS` (or your path).  
Activate env: `conda activate spatialeval_orchestration` (or `spatial_mas`).  
API models require: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

---

## CV-Bench

### Per-model (with_prompt)

| Model | Command |
|-------|---------|
| **Qwen3-4B** | `python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset` |
| **LLaVA4D** | `python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset` |
| **Sa2VA** | `python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset` |
| **GPT-4o** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt` |
| **Claude Sonnet 4.5** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt` |
| **Gemini Robotics-ER** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt` |

### Per-model (without_prompt)

| Model | Command |
|-------|---------|
| **Qwen3-4B** | `python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt` |
| **LLaVA4D** | `python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset --without_prompt` |
| **Sa2VA** | `python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset --without_prompt` |
| **GPT-4o** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt` |
| **Claude Sonnet 4.5** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt` |
| **Gemini Robotics-ER** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt` |

### 3 opensource 같이 (GPU)

| Variant | Command |
|---------|---------|
| **with_prompt** | `python scripts/evals/cvbench/run_all_models_full.py` |
| **without_prompt** | `python scripts/evals/cvbench/run_all_models_full.py --without_prompt` |

### 3 API 같이

| Variant | Command |
|---------|---------|
| **with + without** | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset` |

---

## 3DSRBench

### Per-model (with_prompt)

| Model | Command |
|-------|---------|
| **Qwen3-4B** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset` |
| **LLaVA4D** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset` |
| **Sa2VA** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset` |
| **GPT-4o** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt` |
| **Claude Sonnet 4.5** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt` |
| **Gemini Robotics-ER** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt` |

### Per-model (without_prompt)

| Model | Command |
|-------|---------|
| **Qwen3-4B** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt` |
| **LLaVA4D** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset --without_prompt` |
| **Sa2VA** | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset --without_prompt` |
| **GPT-4o** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt` |
| **Claude Sonnet 4.5** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt` |
| **Gemini Robotics-ER** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt` |

### 3 opensource 같이 (GPU)

| Variant | Command |
|---------|---------|
| **with_prompt** | `python scripts/evals/3dsrbench/run_all_models_full.py` |
| **without_prompt** | Run each model separately with `--without_prompt` (no all-in-one script) |

### 3 API 같이

| Variant | Command |
|---------|---------|
| **with + without** | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset` |

---

## Test (소량 샘플)

| Benchmark | GPU | API |
|-----------|-----|-----|
| CV-Bench | `--max_samples 30` | `--test` |
| 3DSRBench | `--max_samples 30` | `--max_samples 10` |
