# [Fixed Specialist MAS v2] Sa2VA × 3 Roles — CV-Bench Results

## Summary

Fixed Specialist MAS v2 테스트: **Sa2VA**를 3개 role 모두에 고정하여 CV-Bench에서 평가한 결과입니다.

| Metric | Value |
|--------|-------|
| **Overall** | **62.6%** (57/91) |
| counting | 53.3% (32/60) |
| spatial_relation | 80.6% (25/31) |

## Configuration

| Setting | Value |
|---------|-------|
| Specialist | sa2va (고정) |
| Roles | direct_visual_heuristic, explicit_3d_representation, scene_graph_construction |
| Benchmark | CV-Bench |
| Samples | 91 |

## Key Insights

| Category | Accuracy | Notes |
|----------|----------|-------|
| **spatial_relation** | 80.6% | Sa2VA가 공간 관계에서 상대적으로 강함 |
| **counting** | 53.3% | counting 태스크에서 성능 저하 |

## Execution

```bash
# Fixed specialist (sa2va) evaluation
python run_eval_mas_v2.py --benchmark cvbench --specialist sa2va --max_samples 91
```
