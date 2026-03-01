# [Confidence MAS v2] Step-wise Accuracy & Score Evolution — Figures & Analysis

## Summary

Confidence-based MAS v2 was run on **CV-Bench** with **49 samples** (3 specialists: qwen3_4b, llava4d, spatial_reasoner). Step 0 used fixed qwen3_4b for all roles; from step 1 onward, LLMs were selected by category-specific confidence scores updated via `run_step1`. This issue documents the **accuracy trajectory**, **score map evolution**, and **assignment changes** over steps.

**Key findings:**
- **Final accuracy: 87.8%** (43/49)
- **distance_depth**: 100% (17/17) — strongest category
- **spatial_relation**: 81.0% (17/21)
- **counting**: 81.8% (9/11)
- Accuracy rises from 50% (step 2) to peak 89.2% (step 37), then stabilizes around 87–88%
- Final score map converges: qwen3_4b dominates direct_visual for spatial_relation/distance_depth; llava4d for explicit_3d/scene_graph on spatial_relation

## Configuration

| Setting | Value |
|--------|-------|
| Script | `test_confidence_mas_v2.py` |
| Benchmark | CV-Bench |
| Samples | 49 |
| Specialists | qwen3_4b, llava4d, spatial_reasoner |
| Update | `run_step1` (reward-based) |
| Final reasoning | Qwen3-VL-8B (local) |

## Final Results

| Metric | Value |
|--------|-------|
| **Overall** | **87.8%** (43/49) |
| counting | 81.8% (9/11) |
| distance_depth | 100.0% (17/17) |
| spatial_relation | 81.0% (17/21) |

---

## Figure 1: Accuracy over Steps

![Accuracy over steps](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_accuracy_steps.png)

Cumulative accuracy vs step. Starts at 50% (step 2), climbs to ~89% by step 37, then fluctuates slightly; final 87.8%.

---

## Figure 2: Category Distribution

![Category distribution](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_category_dist.png)

Sample count per category in the 49-sample run. spatial_relation and distance_depth dominate; counting has fewer samples.

---

## Figure 3: LLM Assignment Over Steps (by role)

![Assignments](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_assignments.png)

Which LLM was assigned to each role at each step. Early steps show more switching; later steps converge to stable assignments (qwen3_4b for direct_visual, llava4d for explicit_3d/scene_graph on spatial_relation).

---

## Figure 4: Final Confidence Score Heatmap

![Score heatmap](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_score_heatmap.png)

Final score map: category × role × LLM. Green = higher confidence; red = lower. spatial_relation and distance_depth show clear differentiation; size/orientation remain at default 0.5 (no samples in this run).

---

## Figure 5: Per-Category Accuracy

![Category accuracy](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_category_acc.png)

Accuracy by category. distance_depth reaches 100%; spatial_relation and counting ~81%.

---

## Figure 6: Cumulative vs Rolling Accuracy

![Rolling accuracy](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_confidence_rolling_accuracy.png)

Cumulative accuracy (green) vs 5-step rolling average (red). Rolling average smooths fluctuations and shows the upward trend.

---

## Final Optimized Score Map (category × role)

| Category | direct_visual_heuristic | explicit_3d_representation | scene_graph_construction |
|---------|-------------------------|-----------------------------|---------------------------|
| **spatial_relation** | qwen3_4b (1.0) | llava4d (1.0) | llava4d (1.0) |
| **distance_depth** | qwen3_4b (1.0) | qwen3_4b (0.5) | spatial_reasoner (1.0) |
| **counting** | — (0.0) | — (0.0) | qwen3_4b (1.0) |
| size | 0.5 (all) | 0.5 (all) | 0.5 (all) |
| orientation | 0.5 (all) | 0.5 (all) | 0.5 (all) |

---

## Key Insights

| Observation | Implication |
|-------------|-------------|
| Accuracy 50% → 89% → 87.8% | Confidence-based selection improves over fixed assignment; some late-step errors cause slight drop. |
| distance_depth 100% | Best category; qwen3_4b + llava4d + spatial_reasoner combination works well. |
| spatial_relation 81% | More variable; llava4d favored for explicit_3d and scene_graph. |
| counting 81.8% | qwen3_4b dominates scene_graph; direct/explicit scores decayed to 0 (penalized by errors). |
| Assignment convergence | After ~step 20, assignments stabilize; run_step1 successfully differentiates models by category. |

---

## Execution

```bash
python test_confidence_mas_v2.py --benchmark cvbench --max_samples 49
```

Or in Jupyter:

```python
from test_confidence_mas_v2 import run_confidence_mas_test, build_runners_for_confidence

head_gen, spec_gen, reason_gen = build_runners_for_confidence(use_vlm_reasoning=True)
results = run_confidence_mas_test(head_gen, spec_gen, reason_gen, benchmark="cvbench", max_samples=49)
# results["score_history"], results["final_map"]
```

---

## Files

| File | Purpose |
|------|---------|
| `scripts/generate_confidence_mas_figures.py` | Generate figures from parsed log |
| `docs/fig_confidence_*.png` | Output figures |
| `test_confidence_mas_v2.py` | Confidence MAS v2 test script |
