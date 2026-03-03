# Plan: axhub.vaiv.kr에서 SpatialRGPT 전문가로 CV-Bench Count 평가

## 목표

- **파이프라인**: Head (Qwen3-VL-4B) + **3 Specialist 전부 SpatialRGPT** + Final (DeepSeek-R1 / Qwen3-VL-8B)
- **벤치마크**: CV-Bench, **Count 카테고리만**
- **실행 환경**: axhub.vaiv.kr (JupyterHub)
- **결과**: 최종 파이프라인 Accuracy 측정

---

## 가능 여부: ✅ 가능

`test_fixed_specialist_mas_v2.py`가 이미 이 구성을 지원합니다.

- `specialist_whitelist=["spatial_rgpt"]` → 3개 role 모두 SpatialRGPT 사용
- `category_filter=["Count"]` → Count만 평가
- Head / Final은 그대로 유지

---

## 파이프라인 구조

```
[이미지 + 질문]
      │
      ▼
┌─────────────────┐
│  Head Agent     │  Qwen3-VL-4B (카테고리 추론)
│  (Qwen3-VL-4B)  │  → "counting" (Count 전용이면 force_category로 생략 가능)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Specialist 1   │  SpatialRGPT (direct_visual_heuristic)
│  Specialist 2   │  SpatialRGPT (explicit_3d_representation)
│  Specialist 3   │  SpatialRGPT (scene_graph_construction)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Reasoner │  DeepSeek-R1 API 또는 Qwen3-VL-8B (SharedMemory + query → 최종 답)
└────────┬────────┘
         │
         ▼
    [최종 Accuracy]
```

---

## Phase 1: axhub.vaiv.kr 환경 파악

| 항목 | 확인 방법 |
|------|-----------|
| Python | `python --version` (3.10 권장) |
| Conda | `conda --version` |
| CUDA | `nvidia-smi` |
| GPU 메모리 | SpatialRGPT 8B ≈ 16GB+ 필요 |
| 디스크 | 프로젝트 + 모델 캐시용 여유 공간 |

---

## Phase 2: 프로젝트 준비

1. **저장소 클론**
   ```bash
   cd ~
   git clone https://github.com/CY-H1329/Spatial_MAS.git
   cd Spatial_MAS
   ```

2. **SpatialRGPT 서브모듈/폴더**
   - `SpatialRGPT/`가 이미 포함되어 있는지 확인
   - 없으면: `git clone https://github.com/AnjieCheng/SpatialRGPT SpatialRGPT`

3. **데이터**
   - `data/frozen_benchmarks/cvbench_400/` 또는 `cvbench_counting_100/` 존재 확인
   - Count만 쓰려면 `category_filter=["Count"]` 사용

---

## Phase 3: 가상환경 + SpatialRGPT 설치

### 3.1 Conda 환경 생성

```bash
conda create -n srgpt_axhub python=3.10 -y
conda activate srgpt_axhub
pip install --upgrade pip
```

### 3.2 PyTorch (CUDA 버전 확인 후)

```bash
# CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 또는 CUDA 11.8
# pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 SpatialRGPT 의존성

```bash
pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales.git"
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.9.12
pip install sentencepiece shortuuid "pydantic<2" markdown2 requests httpx
pip install "accelerate>=0.27" peft "numpy<2" scikit-learn
pip install opencv-python pillow datasets openai
pip install "transformers>=4.51.0"
```

### 3.4 SpatialRGPT (editable)

```bash
cd SpatialRGPT
pip install -e . --no-deps
cd ..
```

### 3.5 Spatial_MAS 평가용

```bash
pip install huggingface_hub qwen-vl-utils tqdm pyyaml
```

### 3.6 FlashAttention (선택)

- 설치 성공 시: `pip install flash-attn --no-build-isolation`
- 실패 시: `patches/spatialrgpt_flash_attn_fallback.py` 폴백 적용

---

## Phase 4: 실행

### 4.1 환경변수

```bash
export SPATIALRGPT_PATH="$(pwd)/SpatialRGPT"
```

### 4.2 CLI 실행

```bash
python test_fixed_specialist_mas_v2.py \
  --specialist spatial_rgpt \
  --benchmark cvbench \
  --category_filter Count \
  --max_samples 100 \
  --device cuda
```

### 4.3 Jupyter 노트북에서 실행

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path("/home/jovyan/Spatial_MAS")  # axhub 실제 경로로 수정
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.environ["SPATIALRGPT_PATH"] = str(PROJECT_ROOT / "SpatialRGPT")

from test_fixed_specialist_mas_v2 import build_runners_fixed, run_fixed_specialist_mas_test

head_gen, spec_gen, reason_gen = build_runners_fixed(
    specialist="spatial_rgpt",
    specialist_device="cuda",
    use_vlm_reasoning=True,
)

results = run_fixed_specialist_mas_test(
    head_gen, spec_gen, reason_gen,
    specialist="spatial_rgpt",
    benchmark="cvbench",
    max_samples=100,
    category_filter=["Count"],
)

print(f"Accuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
```

---

## Phase 5: 주의사항

| 항목 | 내용 |
|------|------|
| **Final Reasoner** | DeepSeek-R1 API 필요 시 `run_eval_mas_v2.py`의 `reasoning_api_base` 설정. `use_vlm_reasoning=True`면 Qwen3-VL-8B 로컬 사용 |
| **메모리** | Head(4B) + SpatialRGPT(8B) + Final(8B) → 24GB+ VRAM 권장. 부족 시 `specialist_offload_after_use=True` 등으로 CPU 오프로드 |
| **cvbench_400 vs cvbench_counting_100** | `cvbench` + `category_filter=["Count"]` 또는 `benchmark="cvbench_counting_100"` 둘 다 가능 |
| **Count 전용 force_category** | Count만 평가 시 Head Agent 생략, `force_category="counting"` 자동 적용 |

---

## 스크립트 (구현 완료)

| 파일 | 용도 |
|------|------|
| `scripts/setup_axhub_srgpt_env.sh` | axhub용 원스텝 환경 설정 |
| `scripts/run_axhub_srgpt_cvbench_count.sh` | 실행 래퍼 |
| `notebooks/axhub_srgpt_cvbench_count.ipynb` | Jupyter 노트북 실행 |

### axhub에서 실행 순서

```bash
# 1. 클론
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS

# 2. SpatialRGPT 없으면
git clone https://github.com/AnjieCheng/SpatialRGPT SpatialRGPT

# 3. 가상환경 생성
conda create -n srgpt_axhub python=3.10 -y

# 4. 가상환경 활성화
conda activate srgpt_axhub

# 5. 환경설정 (pip install)
bash scripts/setup_axhub_srgpt_env.sh

# 6. 실행
./scripts/run_axhub_srgpt_cvbench_count.sh
```
