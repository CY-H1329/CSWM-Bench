# 3DSRBench & CV-Bench — API Models

Évaluation des modèles API sur **3DSRBench** et **CV-Bench** (`--benchmark cvbench`). **Séparé** du code GPU existant (Qwen3, Sa2VA, LLaVA4D).

- **Push / pull, commandes séparées 3DSRBench vs CV-Bench, où lire les métriques par catégorie** → **[PUSH_PULL_AND_COMMANDS.md](PUSH_PULL_AND_COMMANDS.md)**
- **Par catégorie** : chaque `results.json` contient `per_category_answer_accuracy` (accuracy lettre MCQ par `category` / `task`). `summary.txt` reprend un tableau par modèle ; le terminal imprime aussi `Per-category (answer acc)`.

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

**Référence complète (git + 3DSRBench seul + CV-Bench seul + catégories)** : [PUSH_PULL_AND_COMMANDS.md](PUSH_PULL_AND_COMMANDS.md)

```bash
# --- 3DSRBench seul (défaut : --benchmark 3dsrbench) ---
python scripts/evals/3dsrbench_api/run_eval_api.py
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt_5_2 --prompt_variant with_prompt
bash scripts/evals/3dsrbench_api/h100_run_gpt52_3dsrbench.sh

# --- CV-Bench seul ---
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench
python scripts/evals/3dsrbench_api/run_eval_api.py --benchmark cvbench --full_dataset --model gpt_5_2 --prompt_variant with_prompt
bash scripts/evals/3dsrbench_api/h100_run_gpt52_cvbench.sh

# 50 samples (3DSRBench)
python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 50

# GPT-5.2 seulement, full 3DSRBench
bash scripts/evals/3dsrbench_api/run_3dsrbench_gpt52_full.sh
# ou:
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model gpt_5_2

# Terminaux séparés (GPU / autres modèles) — voir docs/experiments/.../PUSH_PULL_3DSRBench_GPU.md
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
results/runs/cvbench/api_models/<timestamp>/ # --benchmark cvbench
├── claude_sonnet_4_5_with_prompt/
├── claude_sonnet_4_5_without_prompt/
├── gpt4o_with_prompt/
├── gpt4o_without_prompt/
├── gemini_robotics_er_with_prompt/
├── gemini_robotics_er_without_prompt/
├── gpt_5_2_with_prompt/
└── summary.txt
```
