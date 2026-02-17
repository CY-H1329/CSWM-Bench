# Results Structure

This document describes the results folder structure and how to push them to GitHub.

## Raw results (on H100, not pushed)

```
results/
├── runs/
│   └── 3dsrbench/
│       ├── qwen3_4b/
│       │   ├── full_dataset_with_prompt/
│       │   └── full_dataset_without_prompt/
│       ├── llava4d/
│       ├── sa2va/
│       └── api_models/
│           ├── 20260216_121420/    # timestamp
│           │   ├── claude_sonnet_4_5_with_prompt/
│           │   ├── claude_sonnet_4_5_without_prompt/
│           │   ├── gpt4o_with_prompt_1/
│           │   ├── category_claude.csv
│           │   ├── category_gpt4o.csv
│           │   ├── category_gemini.csv
│           │   └── summary.txt
│           └── full_dataset/
```

## Curated results (pushed to GitHub)

`results_summary/` contains aggregated results for the paper. This folder **is** tracked by git.

```
results_summary/
├── 3dsrbench/
│   ├── api_models/
│   │   ├── category_claude.csv
│   │   ├── category_gpt4o.csv
│   │   ├── category_gemini.json
│   │   └── summary.txt
│   └── gpu/
│       └── (aggregated per-model results)
└── README.md
```

## Workflow: H100 → GitHub → Local

### 1. On H100: Gather and push

```bash
cd ~/CY/Spatial_MAS
python scripts/gather_results_summary.py
git add results_summary/
git status
git commit -m "Results: 3DSRBench API + GPU summaries"
git push origin main
```

### 2. On local: Pull and review

```bash
cd ~/Desktop/Spatial_MAS
git pull origin main
# results_summary/ is now updated
```
