# [Specialist] 5 Models × 3 Roles × 2 Benchmarks — 10-Sample Accuracy

## Summary

Each of the **5 specialist VLMs** (Sa2VA, Qwen3-4B, LLaVA4D, SpatialRGPT, SpatialReasoner) is tested across **3 roles** (direct_visual_heuristic, explicit_3d_representation, scene_graph_construction) on **2 benchmarks** (CV-Bench, 3DSRBench) with **10 samples** each. Same prompt per role; only the model changes.

## Configuration

| Setting | Value |
|---------|-------|
| Script | `test_specialist_all_roles.py` |
| Models | sa2va, qwen3_4b, llava4d, spatial_rgpt, spatial_reasoner |
| Roles | direct_visual_heuristic, explicit_3d_representation, scene_graph_construction |
| Benchmarks | CV-Bench, 3DSRBench |
| Samples | 10 per (model, role, benchmark) |
| Environment | conda activate spatial_reasoning |

## Execution

```bash
# Sa2VA
python test_specialist_sa2va_all_roles.py --max_samples 10

# Other models
python test_specialist_all_roles.py --model qwen3_4b --max_samples 10
python test_specialist_all_roles.py --model llava4d --max_samples 10
python test_specialist_all_roles.py --model spatial_rgpt --max_samples 10
python test_specialist_all_roles.py --model spatial_reasoner --max_samples 10
```

---

## Experiment 1: Sa2VA

### Overall Summary

| Role | CV-Bench | 3DSRBench |
|------|----------|-----------|
| direct_visual_heuristic | 7/10 (70.0%) | 6/10 (60.0%) |
| explicit_3d_representation | 7/10 (70.0%) | 5/10 (50.0%) |
| scene_graph_construction | 7/10 (70.0%) | 6/10 (60.0%) |

### Per-Category Detail

**direct_visual_heuristic — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 33.3% | 1/3 |
| Distance | 50.0% | 1/2 |
| Relation | 100.0% | 5/5 |

**direct_visual_heuristic — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 100.0% | 1/1 |
| location_above | 75.0% | 3/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**explicit_3d_representation — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 33.3% | 1/3 |
| Distance | 50.0% | 1/2 |
| Relation | 100.0% | 5/5 |

**explicit_3d_representation — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 100.0% | 1/1 |
| location_above | 75.0% | 3/4 |
| location_closer_to_camera | 0.0% | 0/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**scene_graph_construction — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 33.3% | 1/3 |
| Distance | 100.0% | 2/2 |
| Relation | 80.0% | 4/5 |

**scene_graph_construction — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 100.0% | 1/1 |
| location_above | 75.0% | 3/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

### Sa2VA Key Insights

| Observation | Implication |
|-------------|-------------|
| Relation 80–100% | Sa2VA handles relation-type questions well across roles. |
| Count 33.3% | Counting remains challenging. |
| multi_object_parallel, viewpoint_towards_object 0% | Orientation subcategories are weak (n=1 each). |
| explicit_3d 50% on 3DSRBench | 3D depth tool + Sa2VA underperforms on depth questions. |

---

## Experiment 2: SpatialReasoner

### Overall Summary

| Role | CV-Bench | 3DSRBench |
|------|----------|-----------|
| direct_visual_heuristic | 7/10 (70.0%) | 6/10 (60.0%) |
| explicit_3d_representation | 8/10 (80.0%) | 5/10 (50.0%) |
| scene_graph_construction | 6/10 (60.0%) | 6/10 (60.0%) |

### Per-Category Detail

**direct_visual_heuristic — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 66.7% | 2/3 |
| Distance | 50.0% | 1/2 |
| Relation | 80.0% | 4/5 |

**direct_visual_heuristic — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 100.0% | 4/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**explicit_3d_representation — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 100.0% | 3/3 |
| Distance | 50.0% | 1/2 |
| Relation | 80.0% | 4/5 |

**explicit_3d_representation — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 75.0% | 3/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**scene_graph_construction — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 66.7% | 2/3 |
| Distance | 50.0% | 1/2 |
| Relation | 60.0% | 3/5 |

**scene_graph_construction — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 100.0% | 4/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 0.0% | 0/1 |
| multi_object_parallel | 100.0% | 1/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

### SpatialReasoner Key Insights

| Observation | Implication |
|-------------|-------------|
| explicit_3d CV-Bench 80% | 3D depth tool + SpatialReasoner excels on CV-Bench (best among 2 models so far). |
| Count 100% (explicit_3d) | SpatialReasoner + 3D tool handles counting well on CV-Bench. |
| height_higher 0% | SpatialReasoner struggles on height comparison (n=1). |
| location_above 75–100% | Strong on vertical positioning. |
| multi_object_parallel 100% (scene_graph) | Scene graph helps on parallel alignment. |

---

## Experiment 3: LLaVA4D

### Overall Summary

| Role | CV-Bench | 3DSRBench |
|------|----------|-----------|
| direct_visual_heuristic | 6/10 (60.0%) | 5/10 (50.0%) |
| explicit_3d_representation | 6/10 (60.0%) | 3/10 (30.0%) |
| scene_graph_construction | 5/10 (50.0%) | 4/10 (40.0%) |

### Per-Category Detail

**direct_visual_heuristic — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 0.0% | 0/3 |
| Distance | 50.0% | 1/2 |
| Relation | 100.0% | 5/5 |

**direct_visual_heuristic — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 100.0% | 4/4 |
| location_closer_to_camera | 0.0% | 0/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**explicit_3d_representation — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 0.0% | 0/3 |
| Distance | 50.0% | 1/2 |
| Relation | 100.0% | 5/5 |

**explicit_3d_representation — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 50.0% | 2/4 |
| location_closer_to_camera | 0.0% | 0/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

**scene_graph_construction — CV-Bench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Count | 33.3% | 1/3 |
| Distance | 50.0% | 1/2 |
| Relation | 60.0% | 3/5 |

**scene_graph_construction — 3DSRBench**
| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| height_higher | 0.0% | 0/1 |
| location_above | 50.0% | 2/4 |
| location_closer_to_camera | 50.0% | 1/2 |
| location_next_to | 100.0% | 1/1 |
| multi_object_parallel | 0.0% | 0/1 |
| multi_object_viewpoint_towards_object | 0.0% | 0/1 |

### LLaVA4D Key Insights

| Observation | Implication |
|-------------|-------------|
| Relation 100% (direct_visual, explicit_3d) | LLaVA4D excels on relation-type questions when not using scene graph. |
| Count 0% (direct_visual, explicit_3d) | Counting is very weak; no correct answers on Count category. |
| explicit_3d 3DSRBench 30% | 3D depth tool + LLaVA4D underperforms on 3DSRBench (lowest among 3 models). |
| location_above 100% (direct_visual) vs 50% (explicit_3d, scene_graph) | Direct visual heuristic works best for vertical positioning. |
| height_higher, multi_object_* 0% | Orientation/height subcategories remain challenging. |

---

## Combined Summary (All 5 Models)

| Model | direct_visual_heuristic | explicit_3d_representation | scene_graph_construction |
|-------|-------------------------|----------------------------|---------------------------|
| Sa2VA | CV: 70%, 3D: 60% | CV: 70%, 3D: 50% | CV: 70%, 3D: 60% |
| SpatialReasoner | CV: 70%, 3D: 60% | **CV: 80%**, 3D: 50% | CV: 60%, 3D: 60% |
| LLaVA4D | CV: 60%, 3D: 50% | CV: 60%, 3D: 30% | CV: 50%, 3D: 40% |
| Qwen3-4B | _pending_ | _pending_ | _pending_ |
| SpatialRGPT | _pending_ | _pending_ | _pending_ |

---

## Files

| File | Purpose |
|------|---------|
| `test_specialist_all_roles.py` | Run any model across 3 roles × 2 benchmarks |
| `test_specialist_sa2va_all_roles.py` | Sa2VA-specific wrapper |
| `src2/models/sa2va.py` | Sa2VARunner (with bitsandbytes fallback) |
| `src2/models/llava.py` | LLaVARunner (LLaVA4D: llava-v1.6-mistral-7b-hf) |
| `src2/models/spatial_reasoner.py` | SpatialReasonerRunner |

---

## Next Steps

* Complete Qwen3-4B, SpatialRGPT runs
* Update Combined Summary table
* Create GitHub issue with full results
