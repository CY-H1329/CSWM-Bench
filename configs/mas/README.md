# Spatial MAS Configuration

## Structure

```
configs/mas/
├── config.yaml           # Main config
├── score_table.json      # Per-model per-category weights (initial 0.5)
├── agent_profiles/       # Per-agent profiles
│   ├── qwen3_4b.json
│   ├── sa2va.json
│   ├── llava4d.json
│   ├── claude_sonnet_4_5.json
│   ├── gpt4o.json
│   └── gemini_robotics_er.json
└── README.md
```

## Agent Profile Fields

Each `agent_profiles/<name>.json` contains:

- **name**: Key (e.g. `qwen3_4b`)
- **full_name**: Display name (e.g. `Qwen3-VL-4B`)
- **description**: Short intro for Head-Agent prompt
- **cvbench_per_category**: Count, Relation, Depth, Distance (0–1)
- **3dsrbench_overall**: Overall accuracy
- **3dsrbench_per_category**: 12 categories
- **unified_per_category**: 9 unified categories (depth, distance, relation, …)
- **category_ranking**: Rank 1–6 per category (1 = best)
- **initial_weight**: 0.5
- **runner_type**: `gpu` or `api`
- **model_id**: HuggingFace ID or API model name

## Updating Baselines

After running baseline experiments:

```bash
# CV-Bench: aggregate by category
python scripts/evals/cvbench/aggregate_category_results.py --dir results/runs/cvbench

# 3DSRBench: aggregate by category
python scripts/evals/3dsrbench/aggregate_category_performance.py --dir results/runs/3dsrbench --mode gpu

# Then update agent_profiles/*.json with the new numbers
```

## Score Table

`score_table.json` holds runtime weights. Initial: 0.5 for all. Updated by pipeline:
- Correct: +0.05
- Wrong: -0.02
