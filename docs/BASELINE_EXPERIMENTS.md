# Baseline Experiments

Single-agent evaluation on **3DSRBench** and **CV-Bench**. Multi-agent (Spatial_MAS) will be added in a future release.

---

## Setup

```bash
conda env create -f environment.yml
conda activate spatial_mas
pip install -r requirements.txt
python scripts/setup_datasets.py
```

**GPU**: Qwen3-VL-4B, Sa2VA-4B, LLaVA4D — CUDA, ~16GB+ VRAM. Run one model per terminal.  
**API**: Claude Sonnet 4.5, GPT-4o, Gemini Robotics-ER — set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

**Prompt variants**: `with_prompt` (spatial reasoning STEP 1–4) | `without_prompt` (question + options only).

---

## CV-Bench

### Per-model (with_prompt)

| Model | Command |
|-------|---------|
| Qwen3-4B | `python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset` |
| LLaVA4D | `python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset` |
| Sa2VA | `python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset` |
| GPT-4o | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt` |
| Claude Sonnet 4.5 | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt` |
| Gemini Robotics-ER | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt` |

### Per-model (without_prompt)

| Model | Command |
|-------|---------|
| Qwen3-4B | `python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt` |
| LLaVA4D | `python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset --without_prompt` |
| Sa2VA | `python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset --without_prompt` |
| GPT-4o | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt` |
| Claude Sonnet 4.5 | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt` |
| Gemini Robotics-ER | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt` |

### 3 opensource (GPU)

| Variant | Command |
|---------|---------|
| with_prompt | `python scripts/evals/cvbench/run_all_models_full.py` |
| without_prompt | `python scripts/evals/cvbench/run_all_models_full.py --without_prompt` |

### 3 API

| Variant | Command |
|---------|---------|
| with + without | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset` |

---

## 3DSRBench

### Per-model (with_prompt)

| Model | Command |
|-------|---------|
| Qwen3-4B | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset` |
| LLaVA4D | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset` |
| Sa2VA | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset` |
| GPT-4o | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt` |
| Claude Sonnet 4.5 | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt` |
| Gemini Robotics-ER | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt` |

### Per-model (without_prompt)

| Model | Command |
|-------|---------|
| Qwen3-4B | `python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt` |
| LLaVA4D | `python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset --without_prompt` |
| Sa2VA | `python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset --without_prompt` |
| GPT-4o | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt` |
| Claude Sonnet 4.5 | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt` |
| Gemini Robotics-ER | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt` |

### 3 opensource (GPU)

| Variant | Command |
|---------|---------|
| with_prompt | `python scripts/evals/3dsrbench/run_all_models_full.py` |
| without_prompt | Run each model separately with `--without_prompt` (no all-in-one script) |

### 3 API

| Variant | Command |
|---------|---------|
| with + without | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset` |

---

## Test

| Benchmark | GPU | API |
|-----------|-----|-----|
| 3DSRBench | `--max_samples 30` | `--max_samples 10` |
| CV-Bench | `--max_samples 30` | `--test` |

---

## Output

```
results/runs/
├── 3dsrbench/{qwen3_4b,sa2va,llava4d,api_models}/
└── cvbench/{qwen3_4b,sa2va,llava4d,api_models}/
```
