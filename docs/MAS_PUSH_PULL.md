# Spatial MAS — Push / Pull Workflow (H100)

Full pipeline: Head (GPT-5.2) → 3 Specialists → Reasoning (DeepSeek-VL)

---

## 1. Push (Local → GitHub)

```bash
cd ~/Desktop/Spatial_MAS

git add .
git status
git commit -m "MAS: agent profiles, configs, pipeline"
git push origin main
```

---

## 2. Pull (H100 Server ← GitHub)

```bash
cd /path/to/Spatial_MAS   # e.g. ~/CY/Spatial_MAS

git pull origin main
conda activate spatial_mas   # or spatialeval_orchestration
pip install -r requirements.txt

# Datasets (once)
python scripts/setup_datasets.py
```

---

## 3. Environment Variables (H100)

| Variable | Required | Use |
|---------|----------|-----|
| `OPENAI_API_KEY` | Yes | Head (GPT-5.2), GPT-4o specialist |
| `DEEPSEEK_API_KEY` | No | Reasoning Agent (API fallback; default: GPU open-source) |
| `ANTHROPIC_API_KEY` | Optional | Claude Sonnet specialist |
| `GEMINI_API_KEY` | Optional | Gemini specialist |

GPU specialists (qwen3_4b, sa2va, llava4d) and Reasoning (DeepSeek-VL) run locally on H100 — no API key.

---

## 4. Test (5 samples)

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

# CV-Bench
bash scripts/run_h100_mas_full.sh --test

# Or directly:
python scripts/evals/mas_pipeline/run_eval_mas.py --test
```

---

## 5. Full Run

```bash
# CV-Bench full dataset
bash scripts/run_h100_mas_full.sh --full_dataset

# 3DSRBench full dataset
bash scripts/run_h100_mas_full.sh 3dsrbench --full_dataset

# 50 samples
bash scripts/run_h100_mas_full.sh --max_samples 50
```

---

## 6. Output Structure

```
results/runs/mas_pipeline/
├── test/           # --test
├── full/           # --full_dataset
└── 20260218_123456/   # timestamp
    ├── summary.json
    ├── results.jsonl
    └── progress.json
```

---

## 7. Gather Results & Push

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

python scripts/gather_results_summary.py

git add results_summary/
git status
git commit -m "MAS pipeline results"
git push origin main
```

---

## 8. Config Structure

```
configs/mas/
├── config.yaml
├── score_table.json
└── agent_profiles/
    ├── qwen3_4b.json
    ├── sa2va.json
    ├── llava4d.json
    ├── claude_sonnet_4_5.json
    ├── gpt4o.json
    └── gemini_robotics_er.json
```

Update `agent_profiles/*.json` with real baseline numbers after running:
- `scripts/evals/cvbench/aggregate_category_results.py`
- `scripts/evals/3dsrbench/aggregate_category_performance.py`
