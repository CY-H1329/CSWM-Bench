# MAS + TTO

5 specialists (SpatialRGPT 제외), Trust Score 기반 agent 선택.

## Specialists (5 agents)

- **llava4d** — LLaVA-1.6
- **qwen3_4b** — Qwen3-VL-4B
- **sa2va** — Sa2VA
- **spaceom** — SpaceOm (remyxai/SpaceOm)
- **spatial_reasoner** — SpatialReasoner

## 새 폴더 생성

```bash
cd Spatial_MAS
bash scripts/setup_mas_tto.sh
```

→ `Spatial_MAS_TTO` 폴더가 생성됨

## 실행

```bash
cd Spatial_MAS_TTO
bash scripts/run_mas_tto.sh
```

테스트 (20 samples):

```bash
bash scripts/run_mas_tto.sh --test_only --max_samples 20
```

**GPU 메모리 부족 시** (다른 프로세스와 GPU 공유):

```bash
# 3-agent 모드 (qwen3_4b, llava4d, spatial_reasoner)
LOW_MEMORY=1 bash scripts/run_mas_tto.sh --test_only --max_samples 20

# 또는
bash scripts/run_mas_tto.sh --low_memory --test_only --max_samples 20
```

**4-agent avec SpaceOm** (test, script séparé):

```bash
LOW_MEMORY=1 bash scripts/run_mas_tto_lowmem4.sh --test_only --max_samples 20
# Résultats: results/mas_tto_lowmem4_3dsrbench/
```

## 요구사항

- `data/dataset/3dsrbench_train_300` 존재 (git pull 또는 prepare_train_datasets.py)
- `transformers>=4.45` (SpaceOm용)
- trust_score (spatial_aomas/trust_score.py)
