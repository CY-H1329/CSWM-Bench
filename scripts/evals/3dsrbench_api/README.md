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
| DeepSeek-VL | DeepSeek `/v1/vision` ou OpenRouter | `DEEPSEEK_API_KEY` ou `OPENROUTER_API_KEY` |
| Gemini Robotics-ER | Google GenAI | `GEMINI_API_KEY` |

## Usage

```bash
# 1000 samples (défaut), avec/sans prompt par modèle
python scripts/evals/3dsrbench_api/run_eval_api.py

# 50 samples
python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 50

# Dataset complet → sortie dans full_dataset/
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset
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

## Sorties

```
results/runs/3dsrbench/api_models/<timestamp>/   # ou full_dataset/ si --full_dataset
├── claude_sonnet_4_5_with_prompt/
├── claude_sonnet_4_5_without_prompt/
├── gpt4o_with_prompt/
├── gpt4o_without_prompt/
├── gemini_robotics_er_with_prompt/
├── gemini_robotics_er_without_prompt/
└── summary.txt
```
