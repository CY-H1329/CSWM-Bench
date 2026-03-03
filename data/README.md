# data/

논문 실험용 데이터셋 저장소.

## frozen_benchmarks/

고정된(frozen) 벤치마크 샘플 — **절대 수정하지 마세요.**

| 폴더 | 설명 |
|------|------|
| `cvbench_400/` | CV-Bench 400개 (카테고리당 100개) |
| `3dsrbench_500/` | 3DSRBench 500개 (12 카테고리 균등) |
| `stvqa_full/` | STVQA val 전체 692개 |

재생성: `python scripts/prepare_frozen_benchmarks.py`
