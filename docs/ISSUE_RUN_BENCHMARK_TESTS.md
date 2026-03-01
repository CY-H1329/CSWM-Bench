# [H100/Jupyter] 벤치마크 테스트 실행 가이드

`git pull` 후 Jupyter에서 아래 셀들을 실행하면, 각 에이전트별로 CV-Bench·3DSRBench 성능과 **카테고리별 정확도**를 확인할 수 있습니다.

---

## 1. 사전 준비 (git pull)

```python
import subprocess
subprocess.run(["git", "-C", "/home/jovyan/CY/Spatial_MAS", "pull", "origin", "main"], check=True)
```

---

## 2. Head Agent — 카테고리 분류 테스트

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from test_head_agent_classification import run_classification_test

results = run_classification_test(benchmark="cvbench", max_samples=200)
# 또는 3DSRBench: results = run_classification_test(benchmark="3dsrbench", max_samples=200)
```

---

## 3. 전문가 에이전트별 테스트 (각각 Overall + 카테고리별 정확도)

**핵심**: `from test_specialist_XXX import run_specialist_test` — **import하는 파일만 바꾸면** 다른 전문가를 테스트합니다. 각 파일 안의 `run_specialist_test`가 해당 전문가 전략(tool 포함)을 사용합니다.

| 전문가 | import |
|--------|--------|
| direct_visual_heuristic | `test_specialist_direct_visual` |
| explicit_3d_representation | `test_specialist_explicit_3d` |
| scene_graph_construction | `test_specialist_scene_graph` |

### 3-1. direct_visual_heuristic (pictorial cues, tool 없음)

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_direct_visual import run_specialist_test  # ← direct_visual

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=100)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=100)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

### 3-2. explicit_3d_representation (3D depth tool)

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_explicit_3d import run_specialist_test  # ← explicit_3d

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=50)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=50)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

### 3-3. scene_graph_construction (scene graph tool)

```python
import sys
sys.path.insert(0, "/home/jovyan/CY/Spatial_MAS")

from src2.models.qwen3 import Qwen3Runner
from test_specialist_scene_graph import run_specialist_test  # ← scene_graph

runner = Qwen3Runner(device="cuda")
results_cv = run_specialist_test(runner, benchmark="cvbench", max_samples=50, prefetch_workers=4)
results_3d = run_specialist_test(runner, benchmark="3dsrbench", max_samples=50, prefetch_workers=4)
print(f"CV-Bench:   {results_cv['correct']}/{results_cv['total']} = {100*results_cv['accuracy']:.1f}%")
print(f"3DSRBench:  {results_3d['correct']}/{results_3d['total']} = {100*results_3d['accuracy']:.1f}%")
```

---

## 출력 예시

각 전문가 테스트는 **Overall 정확도**와 **카테고리별 정확도**를 출력합니다:

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

`results['per_category']`로 카테고리별 `correct`/`total`을 확인할 수 있습니다.
