# Frozen Benchmarks (DO NOT MODIFY)

이 폴더의 데이터셋은 **논문 실험용으로 고정된(frozen) 벤치마크**입니다.
어떤 모델로 실험하든 **동일한 샘플**로 평가하기 위해 사용됩니다.

## 벤치마크 구성

| 벤치마크 | 샘플 수 | 샘플링 방식 | 카테고리 |
|----------|---------|-------------|----------|
| **cvbench_400** | 400 | 카테고리당 100개 (4 categories) | Count, Relation, Depth, Distance |
| **3dsrbench_500** | 500 | 12개 카테고리 균등 분배 (~42개/카테고리) | height_higher, location_above, ... |
| **stvqa_full** | 692 | 전체 val split (샘플링 없음) | relation, reach, size, ... |

## 재생성 방법

```bash
python scripts/prepare_frozen_benchmarks.py
```

- Seed: 42 (재현 가능)
- CV-Bench: nyu-visionx/CV-Bench test split
- 3DSRBench: ccvl/3DSRBench benchmark subset
- STVQA: hunarbatra/STVQA-7K val split

## 사용법

로더는 기본적으로 `use_frozen=True`로 frozen 벤치마크를 로드합니다:

```python
from src2.benchmarks.loaders import load_benchmark

ds = load_benchmark("cvbench")  # 400 samples from frozen
ds = load_benchmark("3dsrbench")  # 500 samples from frozen
```

HuggingFace에서 직접 샘플링하려면:

```python
ds = load_benchmark("cvbench", use_frozen=False, max_samples=100)
```

## 주의사항

- **이 데이터셋을 수정하지 마세요.** 논문 실험의 재현성을 위해 동일한 데이터를 유지해야 합니다.
- 데이터가 없으면 `prepare_frozen_benchmarks.py`를 실행하여 생성하세요.
