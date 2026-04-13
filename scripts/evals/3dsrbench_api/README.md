# 3DSRBench — API Models (100 samples)

Évaluation des modèles API sur 3DSRBench. **Séparé** du code GPU existant (Qwen3, Sa2VA, LLaVA4D).

## Dépendances

```bash
pip install anthropic openai google-genai
```

## Modèles

| Modèle | API | Env var |
|--------|-----|---------|
| Claude 3.5 Sonnet | Anthropic | `ANTHROPIC_API_KEY` |
| GPT-4o | OpenAI | `OPENAI_API_KEY` |
| **GPT-5.2** (vision) | OpenAI | `OPENAI_API_KEY` — `config_api.yaml` → `gpt_5_2.model_id` (`gpt-5.2` ou snapshot) |
| DeepSeek-VL | DeepSeek `/v1/vision` ou OpenRouter | `DEEPSEEK_API_KEY` ou `OPENROUTER_API_KEY` |
| Gemini Robotics-ER | Google GenAI | `GEMINI_API_KEY` |

## Usage

```bash
# 1000 samples (défaut), avec/sans prompt par modèle
python scripts/evals/3dsrbench_api/run_eval_api.py

# 50 samples
python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 50

# Dataset complet (split HF `ccvl/3DSRBench` test, **pas** le frozen 500 local)
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset

# GPT-5.2 seulement, full dataset
bash scripts/evals/3dsrbench_api/run_3dsrbench_gpt52_full.sh
# ou:
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt_5_2

# Terminaux séparés (6 runs) — voir PUSH_PULL_3DSRBench_GPU.md
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
```

## Coût estimé (API)

| Dataset | Samples | 3 modèles × 2 variants (with/without prompt) |
|---------|---------|---------------------------------------------|
| 1000 samples (défaut) | 1000 | ~$20–50 (6 runs) |
| Full (~2 700) | ~2 700 | ~$100–300 (6 runs) |

Estimation par requête : ~1 500–2 000 tokens input (image + prompt), ~100 tokens output. Tarifs 2025 : GPT-4o $2.50/1M input, Claude $3/1M input, Gemini moins cher.

## Configuration

- `config_api.yaml` : modèles, API keys (via env), max_samples
- Désactiver un modèle : `enabled: false` dans config
- **DeepSeek-VL** : l’API `api.deepseek.com` rejette `image_url` → le runner utilise `/v1/vision`. Si 404, passer par OpenRouter : `base_url: "https://openrouter.ai/api/v1"`, `model_id: "deepseek/deepseek-vl-7b-chat"`, `api_key_env: "OPENROUTER_API_KEY"`

## Production / 상용 — GPT-5.2, full 3DSRBench

- **Billing**: Paid OpenAI API; terms per your [Business / Enterprise](https://openai.com/enterprise) agreement.
- **Run** (loads `.env` if present):
  ```bash
  bash scripts/evals/3dsrbench_api/run_3dsrbench_gpt52_prod.sh
  ```
- **Retries**: `OPENAI_MAX_RETRIES` (default 8), exponential backoff in `runners.py`.
- **Checkpoint**: `CHECKPOINT_EVERY` env or `--checkpoint_every`; resume with `--resume_dir`, `--start_idx`, `--end_idx`.
- **Hugging Face**: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` if the dataset requires auth.

## Sorties

```
results/runs/3dsrbench/api_models/<timestamp>/   # ou full_dataset/ si --full_dataset
├── claude_sonnet_4_5_with_prompt/
├── claude_sonnet_4_5_without_prompt/
├── gpt4o_with_prompt/
├── gpt4o_without_prompt/
├── gemini_robotics_er_with_prompt/
├── gemini_robotics_er_without_prompt/
├── gpt_5_2_with_prompt/
└── summary.txt
```
