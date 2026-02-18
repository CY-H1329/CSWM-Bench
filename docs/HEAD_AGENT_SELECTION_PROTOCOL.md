# Head-Agent Selection Protocol

This document describes how the Head-Agent was selected and evaluated for the Spatial Multi-Agent System (Spatial_MAS).

---

## 1. Overview

The Head-Agent is the first component in the MAS pipeline. It analyzes the input query and image, then routes the task to downstream agents (Perception, Reasoning). A poorly chosen Head-Agent distorts trust learning and specialization. We therefore evaluate Head-Agent candidates on five core capabilities before deployment.

---

## 2. Head-Agent Candidates

| Model | API | Notes |
|-------|-----|-------|
| GPT-5.2 | OpenAI | Vision, max_completion_tokens |
| Claude Opus 4.5 | Anthropic | Vision |
| GLM-5 | OpenRouter (Zhipu) | Text-only (no vision on OpenRouter) |

---

## 3. Five Core Capabilities

We evaluate each candidate on five capabilities that the Head-Agent must excel at:

| # | Capability | Description | Risk if Poor |
|---|------------|-------------|--------------|
| 1 | **Task Decomposition** | Accurately classify the problem into benchmark categories | Wrong classification distorts trust learning |
| 2 | **Routing Decision** | Choose which agent(s) to invoke (Direct / Perception / Reasoning / Both) | Starting point for specialization learning |
| 3 | **Complexity Estimation** | Judge simple (1) vs complex (5) | Prevents tool overuse and inappropriate shortcuts |
| 4 | **Strategy Planning** | Propose initial tools/strategy | Starting point for Perception policy |
| 5 | **Trust-Aware Logging** | Structure reasoning trace (JSON) | Enables downstream trust updates |

---

## 4. Evaluation Protocol

### 4.1 Benchmarks

- **CV-Bench**: 4 categories (Count, Relation, Depth, Distance)
- **3DSRBench**: 12 fine-grained categories (location_above, height_higher, etc.)

### 4.2 Task Decomposition (Category Routing)

- **Prompt**: Given a question (and optionally image), select the ONE category closest to what the question asks.
- **Metric**: Accuracy vs ground-truth category.
- **Script**: `scripts/evals/head_agent_cvbench/run_eval_category_routing.py`

### 4.3 Routing Decision

- **Prompt**: Given question and category, choose Direct / Perception / Reasoning / Both.
- **Metric**: Format validity (outputs one of the four options).

### 4.4 Complexity Estimation

- **Prompt**: Assign complexity 1–5 with justification.
- **Metric**: Format validity (outputs 1–5).

### 4.5 Strategy Planning

- **Prompt**: Propose 1–3 concrete steps or tools.
- **Metric**: Has numbered steps + relevant keywords (depth, count, relation, etc.).

### 4.6 Trust-Aware Logging

- **Prompt**: Output structured JSON with reasoning, category, route, complexity, confidence.
- **Metric**: Valid JSON + required keys present.

### 4.7 Full Evaluation Script

```bash
python scripts/evals/head_agent_cvbench/run_eval_head_agent_full.py --benchmark all --max_samples 50
```

---

## 5. Selection Criteria

Based on the evaluation results:

1. **Task Decomposition** (category routing accuracy) is the primary criterion — it directly affects downstream trust.
2. **Routing**, **Complexity**, **Strategy**, and **Trust-Logging** scores indicate robustness and usability.
3. Models with consistently high scores across benchmarks and capabilities are preferred.

---

## 6. Results Summary

Results are stored in:

- `results_summary/Spatial_MAS_baseline_and_head_agent.xlsx` — Baseline (SOTA) and Head-Agent selection tables
- `results_summary/HEAD_AGENT_SUMMARY.md` — Auto-generated summary from `summarize_head_agent_results.py`

### Head-Agent Selection (from xlsx)

| Benchmark | Model | Complexity | Routing | Strategy | Task Decomposition | Trust Logging |
|-----------|-------|------------|---------|----------|-------------------|---------------|
| 3DSRBench | GPT-5.2 | 1 | 1 | 0.94 | 0.7 | 1 |
| 3DSRBench | Claude Opus 4.5 | 1 | 1 | 1 | 0.92 | 1 |
| CV-Bench | GPT-5.2 | 1 | 1 | 1 | 1 | 1 |
| CV-Bench | Claude Opus 4.5 | 1 | 1 | 0.99 | 1 | 1 |
| CV-Bench | GLM-5 | 1 | 0.56 | 0.74 | 0.8 | 0.54 |

**Conclusion**: Claude Opus 4.5 and GPT-5.2 perform best across capabilities. GLM-5 (text-only) shows lower scores on routing and trust-logging.

---

## 7. Reproducibility

1. **Environment**: API keys for OpenAI, Anthropic, OpenRouter
2. **Data**: CV-Bench, 3DSRBench (HuggingFace)
3. **Scripts**: `scripts/evals/head_agent_cvbench/`
4. **Gather & Summarize**: `scripts/gather_results_summary.py`, `scripts/summarize_head_agent_results.py`
