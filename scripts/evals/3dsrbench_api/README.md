# 3DSRBench — API Models (100 samples)

Évaluation des modèles API sur 3DSRBench. **Séparé** du code GPU existant (Qwen3, Sa2VA, LLaVA4D).

## Modèles

| Modèle | API | Env var |
|--------|-----|---------|
| Claude 3.5 Sonnet | Anthropic | `ANTHROPIC_API_KEY` |
| GPT-4o | OpenAI | `OPENAI_API_KEY` |
| DeepSeek-VL | DeepSeek (OpenAI-compatible) | `DEEPSEEK_API_KEY` |
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

## Sorties

```
results/runs/3dsrbench/api_models/<timestamp>/
├── claude_3_5_sonnet/
├── gpt4o/
├── deepseek_vl/
├── gemini_robotics_er/
└── summary.txt
```
