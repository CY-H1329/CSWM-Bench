# H100 — MAS v2 환경 호환 (SpatialRGPT / Sa2VA)

SpatialRGPT와 Sa2VA는 구버전 env(Python, transformers, bitsandbytes 등)를 요구할 수 있어, H100 서버의 최신 환경과 충돌하여 5-agent MAS가 동작하지 않을 수 있다.

## 해결 방법

### 1. 3-agent 모드 (권장)

Sa2VA와 SpatialRGPT를 제외하고 **qwen3_4b, llava4d, spatial_reasoner** 3개만 사용한다.

```bash
# CLI
python run_eval_mas_v2.py --benchmark cvbench --use_vlm_reasoning \
  --specialist_whitelist qwen3_4b,llava4d,spatial_reasoner \
  --test_only --max_samples 10

# 또는 스크립트
bash scripts/run_h100_mas_v2.sh 3agent --test_only --max_samples 10
```

### 2. 5-agent 모드 (env 호환 시)

Sa2VA와 SpatialRGPT가 모두 동작하는 환경에서만 사용한다.

```bash
bash scripts/run_h100_mas_v2.sh 5agent --test_only --max_samples 10
```

---

## SpatialRGPT / Sa2VA 이슈 요약

| 모델 | 이슈 | 대응 |
|------|------|------|
| **SpatialRGPT** | VILA/LLaVA 코드베이스, `match` 문법(Python 3.10+), 구버전 transformers | `patch_spatialrgpt_py39.py`로 match→if 변환, 또는 Python 3.10+ 사용 |
| **Sa2VA** | PEFT→bitsandbytes 의존, CUDA 12.x에서 libbitsandbytes_cuda*.so 누락, `_tied_weights_keys` 등 구버전 API | bitsandbytes mock, tied_weights patch, linspace patch (이미 코드에 포함) |

### Sa2VA 추가 시도 (5-agent용)

```bash
# bitsandbytes CUDA 12용 설치
pip install bitsandbytes  # CUDA 12.x 대응 버전

# LD_LIBRARY_PATH 설정 (필요 시)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### SpatialRGPT 추가 시도 (5-agent용)

```bash
# SpatialRGPT 클론 및 패치
git clone https://github.com/AnjieCheng/SpatialRGPT.git
export SPATIALRGPT_PATH=/path/to/SpatialRGPT
python scripts/stvqa7k/patch_spatialrgpt_py39.py  # Python 3.9 사용 시
```

---

## Git pull

```bash
cd /path/to/Spatial_MAS
git remote -v   # remote 확인
git fetch origin
git pull origin main
```

`Repository not found` 또는 권한 오류 시: remote URL, SSH 키, 토큰 확인.
