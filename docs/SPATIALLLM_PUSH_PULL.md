# SpatialLLM 3DSRBench — Push / Pull Workflow (H100)

SpatialReasoner (ccvl/SpatialReasoner) — 3DSRBench 100 샘플 평가

---

## 1. Push (Local → GitHub)

```bash
cd ~/Desktop/Spatial_MAS

git add evals_spatialllm/ docs/SPATIALLLM_PUSH_PULL.md
git status
git commit -m "Add SpatialLLM/SpatialReasoner 3DSRBench eval (100 samples)"
git push origin main
```

---

## 2. Pull (H100 Server ← GitHub)

```bash
cd /path/to/Spatial_MAS   # e.g. ~/CY/Spatial_MAS

git pull origin main
conda activate spatialeval_orchestration   # or spatial_mas

# Datasets (한 번만)
python scripts/setup_datasets.py
```

---

## 3. 실행 (H100)

```bash
cd ~/CY/Spatial_MAS
conda activate spatialeval_orchestration

# 100 샘플 (기본)
bash evals_spatialllm/run_h100.sh

# 또는 직접
python evals_spatialllm/run_spatialllm_3dsrbench.py --max_samples 100
```

**예상 시간**: 100 샘플 기준 약 15–30분 (H100, Qwen2.5-VL 8B)

---

## 4. 결과 Push (H100 → GitHub)

```bash
cd ~/CY/Spatial_MAS

git add results/evals_spatialllm/
git status
git commit -m "SpatialReasoner 3DSRBench results (100 samples)"
git push origin main
```

---

## 5. 출력 구조

```
results/evals_spatialllm/
└── 20260218_HHMMSS/
    ├── summary.json    # {"model_id", "accuracy", "correct", "total"}
    └── results.jsonl   # per-sample: idx, gt, pred, correct
```

---

## 6. 모델 옵션

| --model_id | 설명 |
|------------|------|
| ccvl/SpatialReasoner | SOTA (60.3%) |
| ccvl/SpatialReasoner-SFT | SFT only (58.3%) |
| ccvl/SpatialReasoner-Zero | Zero-shot style (54.0%) |

```bash
python evals_spatialllm/run_spatialllm_3dsrbench.py --model_id ccvl/SpatialReasoner-SFT --max_samples 100
```
