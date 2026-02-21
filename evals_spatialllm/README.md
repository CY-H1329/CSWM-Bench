# SpatialLLM / SpatialReasoner on 3DSRBench

SpatialReasoner (ccvl/SpatialReasoner) — Qwen2.5-VL 기반 3D spatial reasoning 모델. 3DSRBench SOTA.

## 모델

| Model | 3DSRBench (paper) |
|-------|-------------------|
| ccvl/SpatialReasoner | 60.3% |
| ccvl/SpatialReasoner-SFT | 58.3% |
| ccvl/SpatialReasoner-Zero | 54.0% |

## 실행 (H100)

```bash
# 100 샘플 (기본)
bash evals_spatialllm/run_h100.sh

# 50 샘플
bash evals_spatialllm/run_h100.sh 50

# SpatialReasoner-SFT 사용
bash evals_spatialllm/run_h100.sh 100 ccvl/SpatialReasoner-SFT
```

직접 실행:
```bash
python evals_spatialllm/run_spatialllm_3dsrbench.py --max_samples 100
```

## 출력

```
results/evals_spatialllm/
└── 20260218_123456/
    ├── summary.json
    └── results.jsonl
```

## 요구사항

- transformers >= 4.50
- torch, CUDA
- 3DSRBench dataset (ccvl/3DSRBench, HuggingFace cache)
