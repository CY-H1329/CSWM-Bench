# explicit_3d_representation 에이전트 테스트 (50 samples)

서버에서 explicit_3d 에이전트를 3DSRBench, CV-Bench 각 50개씩 테스트하는 방법입니다.

## 사전 준비

```bash
# 1. 프로젝트 폴더로 이동
cd /path/to/Spatial_MAS-main

# 2. git pull (최신 코드)
git pull origin main

# 3. 의존성 (OWL-ViT, DepthAnything, transformers)
pip install transformers torch pillow
# timm (DETR fallback용, OWL-ViT는 transformers에 포함)
pip install timm
```

## 실행

### 3DSRBench 50개 + CV-Bench 50개 (둘 다)

```bash
python test_specialist_explicit_3d.py --benchmark both --max_samples 50 --show_failures 3
```

### 개별 벤치마크

```bash
# CV-Bench 50개
python test_specialist_explicit_3d.py --benchmark cvbench --max_samples 50 --show_failures 3

# 3DSRBench 50개
python test_specialist_explicit_3d.py --benchmark 3dsrbench --max_samples 50 --show_failures 3
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--benchmark` | both | cvbench, 3dsrbench, both |
| `--max_samples` | 50 | 샘플 수 |
| `--seed` | 42 | 랜덤 시드 |
| `--show_failures` | 3 | 오답 상세 출력 개수 |

## 출력 예시

```
============================================================
BENCHMARK: CVBENCH
============================================================
Starting: 50 samples, explicit_3d_representation + Qwen3-VL-4B...
  Progress 10/50 | acc: 80.0%
  Progress 20/50 | acc: 85.0%
  ...
============================================================
SPECIALIST TEST — explicit_3d_representation + Qwen3-VL-4B — CVBENCH
============================================================
Overall: 42/50 = 84.0%

  Count                            62.5%  (5/8)
  Depth                           100.0%  (1/1)
  Distance                        100.0%  (5/5)
  Relation                        100.0%  (6/6)
============================================================

============================================================
BENCHMARK: 3DSRBENCH
============================================================
...
```

## 참고

- **실행 시간**: 샘플당 VLM 2회(객체 추출 + 추론) + OWL-ViT + Depth → 50개 기준 약 5–15분 (GPU)
- **메모리**: Qwen3-VL-4B + OWL-ViT + DepthAnything → 약 8–12GB VRAM
