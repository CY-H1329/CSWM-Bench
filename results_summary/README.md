# Results Summary

<<<<<<< HEAD
Generated: 2026-02-18T09:22:04.434823

Aggregated results for paper submission. Raw data in `results/` on H100.
=======
Aggregated results for paper submission. Raw data remains in `results/` on H100.
>>>>>>> 77adbcf (Head-Agent 5 capabilities, CV-Bench aggregate, 3DSRBench fixes)

## Structure

```
results_summary/
└── 3dsrbench/
    ├── api_models/
    │   └── 20260216_121420/     # Run timestamp
    │       ├── category_claude.csv
    │       ├── category_gpt4o.csv
    │       ├── category_gemini.csv
    │       └── summary.txt
    └── gpu/
        ├── qwen3_4b/
        ├── llava4d/
        └── sa2va/
            └── full_dataset_with_prompt/results.json
```

## Latest API results (20260216_121420)

| Model | Answer Acc | Category Cls Acc |
|-------|------------|------------------|
| claude_sonnet_4_5_with_prompt | 0.33 | 0.44 |
| gpt4o_with_prompt | 0.39 | 0.41 |
| gemini_robotics_er_with_prompt | 0.64 | 0.71 |

## Update

On H100: `python scripts/gather_results_summary.py` then commit and push.
