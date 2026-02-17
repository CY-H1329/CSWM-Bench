# Baseline Experiments

Single-agent baseline evaluation on **3DSRBench** and **CV-Bench**.  
Multi-agent (Spatial_MAS) methodology will be added in a future release.

---

## 1. Experiment Setup

### 1.1 Environment

```bash
conda env create -f environment.yml
conda activate spatial_mas   # or spatialeval_orchestration
pip install -r requirements.txt
python scripts/setup_datasets.py
```

### 1.2 GPU (Open-source models)

- **Qwen3-VL-4B**, **Sa2VA-4B**, **LLaVA4D**
- CUDA, ~16GB+ VRAM recommended
- Run one model per terminal to avoid OOM

### 1.3 API (Closed-source models)

- **Claude Sonnet 4.5**, **GPT-4o**, **Gemini Robotics-ER**
- Set env: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`

### 1.4 Prompt variants

| Variant | Description |
|---------|-------------|
| **with_prompt** | Spatial reasoning prompt (STEP 1–4: classify → plan → reason → answer) |
| **without_prompt** | Question + options only |

---

## 2. Benchmark별 전체 실행 (All-in-one)

### 2.1 3DSRBench — GPU 3 models

```bash
# With prompt (3 models, sequential)
python scripts/evals/3dsrbench/run_all_models_full.py

# Without prompt: run each model separately
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset --without_prompt
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset --without_prompt
```

### 2.2 3DSRBench — API 3 models

```bash
# All 3 models, both variants (6 runs)
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset
```

### 2.3 CV-Bench — GPU 3 models

```bash
# With prompt (3 models, sequential)
python scripts/evals/cvbench/run_all_models_full.py

# Without prompt
python scripts/evals/cvbench/run_all_models_full.py --without_prompt
```

### 2.4 CV-Bench — API 3 models

```bash
# All 3 models, both variants (6 runs)
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset
```

---

## 3. 각 모델 따로 실행 (Per-model)

### 3.1 3DSRBench GPU

```bash
# Qwen3-4B
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt

# Sa2VA-4B
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset --without_prompt

# LLaVA4D
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset --without_prompt
```

### 3.2 3DSRBench API

```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt
```

### 3.3 CV-Bench GPU

```bash
# Qwen3-4B
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt

# Sa2VA-4B
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_sa2va.py --full_dataset --without_prompt

# LLaVA4D
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_llava4d.py --full_dataset --without_prompt
```

### 3.4 CV-Bench API

```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --prompt_variant with_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o --without_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --prompt_variant with_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er --without_prompt
```

---

## 4. Test (소량 샘플)

| Benchmark | GPU test | API test |
|-----------|----------|----------|
| 3DSRBench | `--max_samples 30` | `--max_samples 10` |
| CV-Bench | `--max_samples 30` | `--test` (10 samples) |

---

## 5. Per-dataset summary

| Dataset | GPU models | API models |
|---------|------------|------------|
| 3DSRBench | Qwen3, Sa2VA, LLaVA4D | Claude, GPT-4o, Gemini |
| CV-Bench | Qwen3, Sa2VA, LLaVA4D | Claude, GPT-4o, Gemini |

## 5.1 Per-model summary

| Model | 3DSRBench | CV-Bench |
|-------|-----------|----------|
| Qwen3-4B | GPU | GPU |
| Sa2VA-4B | GPU | GPU |
| LLaVA4D | GPU | GPU |
| Claude Sonnet 4.5 | API | API |
| GPT-4o | API | API |
| Gemini Robotics-ER | API | API |

---

## 6. Output structure

```
results/runs/
├── 3dsrbench/
│   ├── qwen3_4b/
│   ├── sa2va/
│   ├── llava4d/
│   └── api_models/
└── cvbench/
    ├── qwen3_4b/
    ├── sa2va/
    ├── llava4d/
    └── api_models/
```
