# [H100/Jupyter] Per-Agent Benchmark Test Execution Guide

## Summary

This guide provides Jupyter cells to run **individual agent tests** on CV-Bench and 3DSRBench after `git pull`. Each test reports **overall accuracy** and **per-category accuracy**, enabling analysis of which agent excels at which spatial task type.

## Architecture

```
git pull
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Head Agent — Category Classification Test                │
│    test_head_agent_classification.py                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Specialist Agent Tests (3 separate runs)                 │
│    • direct_visual_heuristic  (pictorial cues, no tools)    │
│    • explicit_3d_representation (3D depth tool)              │
│    • scene_graph_construction (scene graph tool)            │
│    Each: test_specialist_XXX.py → CV-Bench + 3DSRBench      │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: Overall accuracy + per_category (correct/total)
```

### Figure 1: Benchmark Test Flow

![Benchmark Test Flow](https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/docs/fig_benchmark_test_flow.png)

## Setup (git pull)

```python
import subprocess
subprocess.run(["git", "-C", "/home/jovyan/CY/Spatial_MAS", "pull", "origin", "main"], check=True)
```

## 1. Head Agent — Category Classification Test

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from test_head_agent_classification import run_classification_test

results = run_classification_test(benchmark="cvbench", max_samples=200)
# Or 3DSRBench: results = run_classification_test(benchmark="3dsrbench", max_samples=200)
```

## 2. Specialist Agent Tests

**Key**: Change only the import — `from test_specialist_XXX import run_specialist_test`. Each module implements that specialist's strategy (with or without tools).

| Specialist | Import | Strategy |
|------------|--------|----------|
| direct_visual_heuristic | `test_specialist_direct_visual` | Pictorial cues (occlusion, size, height). No tools. |
| explicit_3d_representation | `test_specialist_explicit_3d` | 3D depth (VLM + OWL-ViT + DepthAnything). |
| scene_graph_construction | `test_specialist_scene_graph` | 2D relations (VLM + OWL-ViT + pairwise edges). |

### 2.1. direct_visual_heuristic

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_direct_visual import run_specialist_test

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=100)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=100)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

### 2.2. explicit_3d_representation

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_explicit_3d import run_specialist_test

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=50)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=50)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

### 2.3. scene_graph_construction

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_scene_graph import run_specialist_test

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=50, prefetch_workers=4)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=50, prefetch_workers=4)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

## Output Example

Each specialist test prints **overall accuracy** and **per-category accuracy**:

```
SPECIALIST TEST — direct_visual_heuristic + Qwen3-VL-4B — CVBENCH
============================================================
Overall: 812/961 = 84.5%

  Count                            70.7%  (181/256)
  Depth                            91.9%  (203/221)
  Distance                         88.1%  (193/219)
  Relation                         88.7%  (235/265)
============================================================
```

Use `results['per_category']` for programmatic access to per-category `correct`/`total`.

## Files

| File | Purpose |
|------|---------|
| `test_head_agent_classification.py` | Head Agent category classification |
| `test_specialist_direct_visual.py` | direct_visual_heuristic |
| `test_specialist_explicit_3d.py` | explicit_3d_representation |
| `test_specialist_scene_graph.py` | scene_graph_construction |
