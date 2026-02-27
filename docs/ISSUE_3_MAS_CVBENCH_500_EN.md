# [MAS v2] CV-Bench 500-Sample Final Performance Results

## Summary

Full MAS v2 pipeline (Head → 3 Specialists → Final Reasoning Agent with Qwen3-VL-8B image+text) evaluated on **500 CV-Bench samples** (480 valid). Achieves **86.7% overall accuracy** with strong performance on distance_depth (93.5%) and spatial_relation (91.0%); counting remains the hardest category (66.3%).

## Configuration

| Setting | Value |
|---------|-------|
| Pipeline | Head Agent → 3 Specialists → Final Reasoning (Qwen3-VL-8B, image+text) |
| Benchmark | CV-Bench |
| Samples | 500 requested, 480 valid |
| Optimization | `shared_object_extraction=True`, `prefetch_workers=4` |

## Execution Code

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

## Results

### Overall

| Metric | Value |
|--------|-------|
| **Accuracy** | **86.7%** |
| Correct / Total | 416 / 480 |

### Per-Category

| Category | Accuracy | Correct / Total |
|----------|----------|-----------------|
| counting | 66.3% | 65 / 98 |
| distance_depth | 93.5% | 159 / 170 |
| orientation | 0.0% | 0 / 1 |
| spatial_relation | 91.0% | 192 / 211 |

### Figure 1: Per-Category Accuracy

![MAS CV-Bench 500 Results](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_mas_cvbench_500_results.png)

## Progress Log

```
  Progress 10/480  | acc: 70.0%
  Progress 100/480 | acc: 78.0%
  Progress 200/480 | acc: 84.0%
  Progress 300/480 | acc: 87.7%
  Progress 400/480 | acc: 87.1%
  Progress 480/480 | acc: 86.7%
```

(Full log available in `docs/ISSUE_MAS_CVBENCH_500_RESULTS.md`.)

## Key Insights

| Observation | Implication |
|-------------|-------------|
| distance_depth 93.5% | 3D/depth specialists + Final Reasoning with image work well for depth questions. |
| spatial_relation 91.0% | Scene graph + direct visual support relation tasks effectively. |
| counting 66.3% | Counting remains challenging; may need stronger Count Protocol or specialist tuning. |
| orientation 0/1 | Single sample; not statistically meaningful. |

## Next Steps

* Compare with DeepSeek-R1 (text-only) Final Reasoning on same 500 samples
* Run 3DSRBench 500-sample evaluation
* Ablation: specialist contribution analysis
