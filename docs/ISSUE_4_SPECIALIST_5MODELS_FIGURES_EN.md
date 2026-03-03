# [Specialist] 5 Models × 3 Roles × 2 Benchmarks — Figures & Analysis

## Summary

Five specialist VLMs (Qwen3-4B, Sa2VA, SpatialReasoner, LLaVA4D, SpatialRGPT) were evaluated across **3 roles** (direct_visual_heuristic, explicit_3d_representation, scene_graph_construction) on **2 benchmarks** (CV-Bench, 3DSRBench) with **10 samples** each. This issue summarizes results with figures and task/benchmark tendencies.

**Key findings:**
- **Qwen3-4B** dominates CV-Bench (77–89%): strongest on Count, Distance
- **SpatialRGPT** best on 3DSRBench direct_visual (60%): image-only works well for 3D
- **LLaVA4D** weakest overall; explicit_3d on 3DSRBench drops to 30%
- **multi_object_*** tasks remain challenging across all models (0–50%)

## Configuration

| Setting | Value |
|---------|-------|
| Script | `test_specialist_all_roles.py` |
| Models | qwen3_4b, sa2va, spatial_reasoner, llava4d, spatial_rgpt |
| Roles | direct_visual_heuristic, explicit_3d_representation, scene_graph_construction |
| Benchmarks | CV-Bench, 3DSRBench |
| Samples | 10 per (model, role, benchmark) |

## Combined Results Table

| Model | direct_visual_heuristic | explicit_3d_representation | scene_graph_construction |
|-------|-------------------------|----------------------------|---------------------------|
| **Qwen3-4B** | **CV: 89%, 3D: 50%** | **CV: 78%, 3D: 50%** | **CV: 89%, 3D: 40%** |
| Sa2VA | CV: 70%, 3D: 60% | CV: 70%, 3D: 50% | CV: 70%, 3D: 60% |
| SpatialReasoner | CV: 70%, 3D: 60% | CV: 80%, 3D: 50% | CV: 60%, 3D: 60% |
| LLaVA4D | CV: 60%, 3D: 50% | CV: 60%, 3D: 30% | CV: 50%, 3D: 40% |
| SpatialRGPT | CV: 50%, 3D: 60% | CV: 60%, 3D: 40% | CV: 60%, 3D: 50% |

---

## Figure 1: CV-Bench vs 3DSRBench (avg across roles)

![CV vs 3D](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_specialist_cv_vs_3d.png)

Qwen3-4B leads on CV-Bench; SpatialRGPT and Sa2VA are stronger on 3DSRBench.

---

## Figure 2: Accuracy Heatmap

![Heatmap](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_specialist_heatmap.png)

Model × (Role × Benchmark) accuracy. Green = higher, Red = lower.

---

## Figure 3: Model Strengths (Best per Role × Benchmark)

![Strengths](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_specialist_strengths.png)

Number of (role, benchmark) combinations where each model achieves the highest accuracy.

---

## Figure 4: Accuracy by Role

![By Role](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_specialist_by_role.png)

CV-Bench vs 3DSRBench for each role. Direct visualization is strongest for Qwen3; explicit_3d helps SpatialReasoner on CV-Bench.

---

## Figure 5: Task & Benchmark Tendencies

![Tendencies](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_specialist_tendencies.png)

---

## Summary: Task Strengths & Benchmarks

| Task / Benchmark | Strong Model | Weak Model |
|------------------|-------------|------------|
| **CV-Bench (overall)** | Qwen3-4B (77–89%) | LLaVA4D (50–60%) |
| **3DSRBench (overall)** | SpatialRGPT direct (60%), Sa2VA (60%) | LLaVA4D explicit_3d (30%) |
| **Count** | Qwen3-4B (100%) | LLaVA4D (0%) |
| **Relation** | Sa2VA (80–100%) | Qwen3 (67%) |
| **Distance** | Qwen3 (100% on CV) | LLaVA4D (50%) |
| **location_above** | SpatialRGPT, SpatialReasoner (100%) | — |
| **multi_object_*** | All models struggle (0–50%) | — |

---

## Key Insights

| Observation | Implication |
|-------------|-------------|
| Qwen3 77–89% on CV-Bench | Best general-purpose specialist for 2D spatial questions. |
| SpatialRGPT 60% on 3DSRBench direct | Image-only mode works well for 3D; no need for depth/region tools in this setup. |
| LLaVA4D explicit_3d 30% on 3DSRBench | 3D depth tool + LLaVA4D underperforms; may need different tooling. |
| SpatialReasoner explicit_3d 80% on CV | 3D depth tool + SpatialReasoner excels on CV-Bench. |
| Count: Qwen3 100% vs LLaVA4D 0% | Large variance in counting ability across models. |
| multi_object_* 0–50% | Multi-object spatial reasoning remains an open challenge. |

---

## Execution

```bash
python test_specialist_all_roles.py --model qwen3_4b --max_samples 10
python test_specialist_all_roles.py --model sa2va --max_samples 10
python test_specialist_all_roles.py --model spatial_reasoner --max_samples 10
python test_specialist_all_roles.py --model llava4d --max_samples 10
python test_specialist_all_roles.py --model spatial_rgpt --max_samples 10
```

---

## Files

| File | Purpose |
|------|---------|
| `scripts/generate_specialist_figures.py` | Generate figures from results |
| `docs/fig_specialist_*.png` | Output figures |
| `docs/ISSUE_SPECIALIST_5MODELS_10SAMPLES.md` | Full per-category results |
