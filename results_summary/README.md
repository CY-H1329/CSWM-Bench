# Results Summary

Aggregated results for paper submission. Raw data remains in `results/` on H100.

## Structure

- `3dsrbench/api_models/` — Claude, GPT-4o, Gemini (category CSV/JSON, summary.txt)
- `3dsrbench/gpu/` — Qwen3, Sa2VA, LLaVA4D (results.json per run)

## How to update

On H100:

```bash
cd ~/CY/Spatial_MAS
python scripts/gather_results_summary.py
git add results_summary/
git commit -m "Results: 3DSRBench summaries"
git push origin main
```
