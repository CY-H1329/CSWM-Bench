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
# 100 samples (défaut)
python scripts/evals/3dsrbench_api/run_eval_api.py

# 50 samples
python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 50
```

## Configuration

- `config_api.yaml` : modèles, API keys (via env), max_samples
- Désactiver un modèle : `enabled: false` dans config
- **DeepSeek-VL** : l’API `api.deepseek.com` rejette `image_url` → le runner utilise `/v1/vision`. Si 404, passer par OpenRouter : `base_url: "https://openrouter.ai/api/v1"`, `model_id: "deepseek/deepseek-vl-7b-chat"`, `api_key_env: "OPENROUTER_API_KEY"`

## Sorties

```
results/runs/3dsrbench/api_models/<timestamp>/
├── claude_sonnet_4_5/
├── gpt4o/
├── deepseek_vl/
├── gemini_robotics_er/
└── summary.txt
```
