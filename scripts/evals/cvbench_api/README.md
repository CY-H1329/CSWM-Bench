# CV-Bench API Models

Claude Sonnet 4.5, GPT-4o, Gemini Robotics-ER (via API).

## Env

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY
```

---

## 1. Test (무료 API) — 10 samples

모든 모델, with/without prompt. API 연결 확인용.

```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

python scripts/evals/cvbench_api/run_eval_api.py --test
```

Output: `results/runs/cvbench/api_models/test/`

---

## 2. Full dataset — ~2638 samples

6 runs (3 models × 2 prompt variants). 터미널 분리 권장.

### Option A: 한 번에 (순차)

```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset
```

### Option B: 모델별 터미널 (병렬)

**Terminal 1 — Claude**
```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5
```

**Terminal 2 — GPT-4o**
```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gpt4o
```

**Terminal 3 — Gemini**
```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model gemini_robotics_er
```

### Option C: 단일 run (with/without prompt)

```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
```

---

## Output

```
results/runs/cvbench/api_models/
├── test/                    # --test (10 samples)
│   ├── claude_sonnet_4_5_with_prompt/
│   ├── claude_sonnet_4_5_without_prompt/
│   ├── gpt4o_with_prompt/
│   ├── gpt4o_without_prompt/
│   ├── gemini_robotics_er_with_prompt/
│   ├── gemini_robotics_er_without_prompt/
│   └── summary.txt
└── full_dataset/            # --full_dataset
    └── ...
```
