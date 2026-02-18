# Results Summary

Generated: 2026-02-18

Aggregated results for paper submission. Raw data remains in `results/` on H100.

## Structure

- `3dsrbench/api_models/` — Claude, GPT-4o, Gemini (category CSV, summary)
- `3dsrbench/gpu/` — Qwen3, Sa2VA, LLaVA4D (results.json per run)
- `cvbench/gpu/` — Qwen3, Sa2VA, LLaVA4D (results.json per run)
- `cvbench/api_models/` — Claude, GPT-4o, Gemini (results.json, summary.txt)
- `head_agent/cvbench/`, `head_agent/3dsrbench/` — Head-Agent category routing (GPT-5.2, Claude Opus 4.5, GLM-5)

## Update

On H100: `python scripts/gather_results_summary.py` then commit and push.
