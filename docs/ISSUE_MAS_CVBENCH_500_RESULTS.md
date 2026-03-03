# [MAS v2] CV-Bench 500샘플 최종 성능 결과

## 설정

- **파이프라인**: Head Agent → 3 Specialists → Final Reasoning Agent (Qwen3-VL-8B, image+text)
- **벤치마크**: CV-Bench
- **샘플 수**: 500 (유효 480)
- **최적화**: `shared_object_extraction=True`, `prefetch_workers=4`

## 실행 코드

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from run_eval_mas_v2 import build_runners
from test_final_reasoning_mas_v2 import run_mas_test

head_gen, spec_gen, reason_gen = build_runners(
    specialist_device="cuda",
    use_vlm_reasoning=True,
    reasoning_vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
)

results = run_mas_test(
    head_gen, spec_gen, reason_gen,
    benchmark="cvbench",
    max_samples=500,
    prefetch_workers=4,
    use_vlm_reasoning=True,
)
print(f"CV-Bench: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
```

## 결과

### Overall

| 항목 | 값 |
|------|-----|
| **정확도** | **86.7%** |
| **정답/전체** | 416 / 480 |

### 카테고리별

| 카테고리 | 정확도 | 정답/전체 |
|----------|--------|-----------|
| counting | 66.3% | 65/98 |
| distance_depth | 93.5% | 159/170 |
| orientation | 0.0% | 0/1 |
| spatial_relation | 91.0% | 192/211 |

### 진행 로그 (Progress)

```
  Progress 10/480  | acc: 70.0%
  Progress 20/480  | acc: 75.0%
  Progress 30/480  | acc: 73.3%
  Progress 40/480  | acc: 77.5%
  Progress 50/480  | acc: 80.0%
  Progress 60/480  | acc: 78.3%
  Progress 70/480  | acc: 75.7%
  Progress 80/480  | acc: 76.2%
  Progress 90/480  | acc: 76.7%
  Progress 100/480 | acc: 78.0%
  Progress 110/480 | acc: 78.2%
  Progress 120/480 | acc: 79.2%
  Progress 130/480 | acc: 80.0%
  Progress 140/480 | acc: 81.4%
  Progress 150/480 | acc: 80.7%
  Progress 160/480 | acc: 81.2%
  Progress 170/480 | acc: 82.4%
  Progress 180/480 | acc: 83.3%
  Progress 190/480 | acc: 84.2%
  Progress 200/480 | acc: 84.0%
  Progress 210/480 | acc: 84.8%
  Progress 220/480 | acc: 85.5%
  Progress 230/480 | acc: 86.1%
  Progress 240/480 | acc: 86.7%
  Progress 250/480 | acc: 86.8%
  Progress 260/480 | acc: 87.3%
  Progress 270/480 | acc: 87.4%
  Progress 280/480 | acc: 87.5%
  Progress 290/480 | acc: 87.2%
  Progress 300/480 | acc: 87.7%
  Progress 310/480 | acc: 87.4%
  Progress 320/480 | acc: 87.2%
  Progress 330/480 | acc: 86.7%
  Progress 340/480 | acc: 86.8%
  Progress 410/480 | acc: 87.1%
  Progress 420/480 | acc: 87.4%
  Progress 430/480 | acc: 87.2%
  Progress 440/480 | acc: 87.0%
  Progress 450/480 | acc: 86.7%
  Progress 460/480 | acc: 86.7%
  Progress 470/480 | acc: 86.8%
  Progress 480/480 | acc: 86.7%
```
