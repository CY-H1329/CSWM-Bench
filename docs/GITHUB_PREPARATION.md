# GitHub Preparation Guide

이 문서는 Spatial_MAS 코드를 GitHub에 올리기 전 **어떤 파일을 포함하고, 어떤 코멘트를 추가해야 하는지** 정리합니다.

---

## 1. 포함할 파일 (Track)

### 1.1 핵심 코드

| 경로 | 설명 |
|------|------|
| `config.yaml` | 메인 설정 (benchmark, models, eval) |
| `run_eval_mas.py` | MAS 파이프라인 |
| `run_eval_mas_full.py` | MAS full |
| `run_eval_single_3dsrbench.py` | 3DSRBench 단일 실행 |
| `environment.yml` | Conda 환경 |
| `requirements.txt` | pip 의존성 (있다면) |

### 1.2 3DSRBench

| 경로 | 설명 |
|------|------|
| `scripts/evals/3dsrbench/common.py` | Spatial prompt (12 categories) |
| `scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py` | Qwen3-4B GPU |
| `scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py` | Sa2VA-4B GPU |
| `scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py` | LLaVA4D GPU |
| `scripts/evals/3dsrbench/run_all_models_full.py` | 3개 모델 일괄 |
| `scripts/evals/3dsrbench/aggregate_category_performance.py` | 카테고리별 집계 |
| `scripts/evals/3dsrbench/compare_predictions.py` | 예측 비교 |
| `scripts/evals/3dsrbench/README.md` | 3DSRBench 명령어 |
| `scripts/evals/3dsrbench_api/run_eval_api.py` | API 모델 (Claude, GPT-4o, Gemini) |
| `scripts/evals/3dsrbench_api/config_api.yaml` | API 설정 |
| `scripts/evals/3dsrbench_api/runners.py` | API 클라이언트 |
| `scripts/evals/3dsrbench_api/README.md` | API 명령어 |

### 1.3 CV-Bench

| 경로 | 설명 |
|------|------|
| `scripts/evals/cvbench/common.py` | Spatial prompt (Count, Relation, Depth, Distance) |
| `scripts/evals/cvbench/explore_categories.py` | Task 타입 탐색 |
| `scripts/evals/cvbench/run_eval_cvbench_qwen3.py` | Qwen3-4B GPU |
| `scripts/evals/cvbench/run_eval_cvbench_sa2va.py` | Sa2VA-4B GPU |
| `scripts/evals/cvbench/run_eval_cvbench_llava4d.py` | LLaVA4D GPU |
| `scripts/evals/cvbench/README.md` | CV-Bench GPU 명령어 |
| `scripts/evals/cvbench_api/run_eval_api.py` | API 모델 |
| `scripts/evals/cvbench_api/config_api.yaml` | API 설정 |
| `scripts/evals/cvbench_api/README.md` | CV-Bench API 명령어 |

### 1.4 공통

| 경로 | 설명 |
|------|------|
| `src/benchmarks/loaders.py` | 3DSRBench, CV-Bench 로더 |
| `src/data.py` | normalize_answer, accuracy, extract_predicted_category |
| `src/models/*.py` | Qwen3, Sa2VA, LLaVA |
| `scripts/setup_datasets.py` | 데이터셋 다운로드 |
| `scripts/gather_results_summary.py` | 결과 수집 |
| `results_summary/` | 집계된 결과 (versioned) |

### 1.5 문서

| 경로 | 설명 |
|------|------|
| `docs/*.md` | 전체 문서 |
| `docs/experiments/baseline_experiments/single_agent/PUSH_PULL_3DSRBench_GPU.md` | 3DSRBench Push/Pull |
| `docs/experiments/baseline_experiments/single_agent/PUSH_PULL_CVBench.md` | CV-Bench Push/Pull |

---

## 2. 제외할 파일 (.gitignore)

```
results/           # 원본 결과 (용량 큼)
*.pyc
__pycache__/
.conda/
.env               # API 키 절대 포함 금지
.DS_Store
```

**주의**: `results_summary/`는 **포함** (논문용 집계 결과).

---

## 3. 코드 코멘트 체크리스트

### 3.1 각 eval 스크립트 상단

```python
#!/usr/bin/env python3
"""
[Benchmark] evaluation — [Model] only.
Full dataset, with/without spatial prompt.

Usage:
  python scripts/evals/[benchmark]/run_eval_[benchmark]_[model].py --max_samples 30
  python scripts/evals/[benchmark]/run_eval_[benchmark]_[model].py --full_dataset
  python scripts/evals/[benchmark]/run_eval_[benchmark]_[model].py --full_dataset --without_prompt

Requirements: CUDA, conda env spatial_mas
"""
```

### 3.2 config 파일

```yaml
# config.yaml
# Spatial_MAS evaluation config
# Override: CUDA_VISIBLE_DEVICES, model paths, API keys (env)

dataset:
  benchmark: "3dsrbench"    # 3dsrbench, cvbench
  split: "test"
  max_samples: null         # null = all
```

### 3.3 API config

```yaml
# config_api.yaml
# API keys via env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

dataset:
  test_samples: 10      # --test 시 사용
  max_samples: 2638     # CV-Bench full
```

### 3.4 common.py (prompt)

```python
"""
Shared logic for [Benchmark] single-model evaluation.
Task categories: [list]. Agent infers category by itself (STEP 1).
"""
```

### 3.5 loaders.py

```python
"""
Unified loaders for benchmarks.
Returns: image, question, options, answer, category.
Benchmarks: 3DSRBench, CV-Bench
"""
```

---

## 4. 3DSRBench 실험 — GitHub용 정리

### 4.1 포함 파일

```
scripts/evals/3dsrbench/
├── common.py
├── run_eval_3dsrbench_qwen3.py
├── run_eval_3dsrbench_sa2va.py
├── run_eval_3dsrbench_llava4d.py
├── run_all_models_full.py
├── aggregate_category_performance.py
├── compare_predictions.py
└── README.md

scripts/evals/3dsrbench_api/
├── run_eval_api.py
├── config_api.yaml
├── runners.py
├── recover_from_responses.py
└── README.md
```

### 4.2 README에 넣을 명령어

**GPU (30 samples 테스트):**
```bash
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 30
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 30 --without_prompt
```

**GPU (full):**
```bash
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset --without_prompt
# Sa2VA, LLaVA4D 동일
```

**API (10 samples 테스트):**
```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --max_samples 10 --model claude_sonnet_4_5 --prompt_variant with_prompt
```

**API (full):**
```bash
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
```

---

## 5. CV-Bench 실험 — GitHub용 정리

### 5.1 포함 파일

```
scripts/evals/cvbench/
├── common.py
├── explore_categories.py
├── run_eval_cvbench_qwen3.py
├── run_eval_cvbench_sa2va.py
├── run_eval_cvbench_llava4d.py
└── README.md

scripts/evals/cvbench_api/
├── run_eval_api.py
├── config_api.yaml
└── README.md
```

### 5.2 README에 넣을 명령어

**Task 탐색:**
```bash
python scripts/evals/cvbench/explore_categories.py --max_samples 500
```

**GPU (30 samples 테스트):**
```bash
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --max_samples 30
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --max_samples 30 --without_prompt
```

**GPU (full):**
```bash
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset
python scripts/evals/cvbench/run_eval_cvbench_qwen3.py --full_dataset --without_prompt
```

**API (10 samples 테스트):**
```bash
python scripts/evals/cvbench_api/run_eval_api.py --test --model claude_sonnet_4_5 --prompt_variant with_prompt
python scripts/evals/cvbench_api/run_eval_api.py --test --model claude_sonnet_4_5 --without_prompt
```

**API (full):**
```bash
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --prompt_variant with_prompt
python scripts/evals/cvbench_api/run_eval_api.py --full_dataset --model claude_sonnet_4_5 --without_prompt
# gpt4o, gemini_robotics_er 동일
```

---

## 6. Commit 전 체크리스트

- [ ] `.env` 또는 API 키가 코드/설정에 포함되지 않음
- [ ] `results/` 가 .gitignore에 있음
- [ ] `results_summary/` 최신 결과 포함 (선택)
- [ ] 각 스크립트 상단에 Usage 코멘트 있음
- [ ] README에 테스트/풀 실행 명령어 정리됨
- [ ] `python scripts/setup_datasets.py` 로 데이터셋 준비 가능

---

## 7. Push 명령어

```bash
cd ~/Desktop/Spatial_MAS   # 또는 ~/CY/Spatial_MAS

git status
git add .
git add -u
git status
git commit -m "Add 3DSRBench + CV-Bench eval scripts, docs"
git push origin main
```

**결과만 push:**
```bash
python scripts/gather_results_summary.py
git add results_summary/
git commit -m "Results: 3DSRBench + CV-Bench summaries"
git push origin main
```
