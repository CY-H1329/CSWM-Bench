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

## All-in-one

| Benchmark | GPU | API |
|-----------|-----|-----|
| 3DSRBench | `python scripts/evals/3dsrbench/run_all_models_full.py` | `python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset` |
| CV-Bench | `python scripts/evals/cvbench/run_all_models_full.py` | `python scripts/evals/cvbench_api/run_eval_api.py --full_dataset` |

For without_prompt: add `--without_prompt` (CV-Bench GPU). For 3DSRBench GPU without_prompt, run each model separately.

---

## Per-model

**GPU**: `run_eval_3dsrbench_*` / `run_eval_cvbench_*` — add `--without_prompt` for variant.  
**API**: `run_eval_api.py --full_dataset --model {claude_sonnet_4_5,gpt4o,gemini_robotics_er}` — add `--prompt_variant with_prompt` or `--without_prompt`.

See [COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md) for full list.

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
