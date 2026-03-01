# SpatialRGPT + datasets 호환성 상세 분석

## 1. 모델 vs 데이터 로딩 — 혼동 금지

**SpatialRGPT 모델은 정상 로드됩니다.**

로그에서 확인:
```
Loading checkpoint shards: 100%|████| 4/4
Resuming region extractor from: .../SpatialRGPT-VILA1.5-8B/region_extractor
```

- **롤 프롬프트**: 가능 (모델이 로드된 후 사용)
- **이미지 입력**: 가능 (SpatialRGPT는 VLM)
- **실제 오류**: **벤치마크 데이터셋 로딩 단계**에서 발생 (모델 추론 전)

---

## 2. 오류 발생 위치

```
load_benchmark("cvbench", ...)
  → load_from_disk (frozen) 실패
  → load_dataset("nyu-visionx/CV-Bench") fallback
  → DatasetInfo.from_directory(cache_dir)
  → Features.from_dict(features)
  → TypeError: must be called with a dataclass type or instance
```

**에러는 `datasets` 라이브러리가 캐시된 `dataset_info.json`을 파싱할 때 발생합니다.**

---

## 3. 근본 원인

| 항목 | 내용 |
|------|------|
| **srgpt 환경** | `datasets==2.16.1` (SpatialRGPT pyproject) |
| **캐시 생성** | `spatial_reasoning` 또는 더 최신 `datasets`로 생성됨 |
| **캐시 경로** | `~/.cache/huggingface/datasets/` |
| **문제** | 2.16.1이 새 형식의 Features를 파싱하지 못함 |

`Features.from_dict()`가 중첩된 feature 정의(예: `List[Value]`, `Image`)를 처리할 때, 2.16.1과 호환되지 않는 형식이 들어와 `dataclasses.fields()`에서 실패합니다.

---

## 4. 해결 방법

### 방법 A: srgpt에서 datasets 업그레이드 (권장)

```bash
conda activate srgpt
pip install "datasets>=2.18"
```

SpatialRGPT 학습에 영향이 있을 수 있으므로, **추론만** 할 경우 시도해볼 만합니다.

### 방법 B: CV-Bench 캐시 삭제 후 재다운로드

```bash
# CV-Bench 캐시 삭제
rm -rf ~/.cache/huggingface/datasets/nyu-visionx___cv-bench

# 또는 전체 datasets 캐시
rm -rf ~/.cache/huggingface/datasets/
```

이후 `load_dataset`을 다시 실행하면 srgpt의 `datasets` 2.16.1로 새 캐시가 생성됩니다.  
단, HuggingFace에 올라간 CV-Bench 메타데이터 자체가 새 형식이면 여전히 실패할 수 있습니다.

### 방법 C: spatial_reasoning 환경에서 SpatialRGPT 실행

Python 3.9 + 패치 방식으로 SpatialRGPT를 `spatial_reasoning`에서 실행하면,  
`datasets` 버전이 맞아서 frozen/캐시 로딩이 정상 동작합니다.

---

## 5. 요약

| 질문 | 답변 |
|------|------|
| 모델이 롤 프롬프트를 못 하나? | **가능함** — 오류는 데이터 로딩 단계에서 발생 |
| 이미지를 못 받나? | **받을 수 있음** — VLM으로 이미지 입력 지원 |
| 실제 문제는? | **datasets 버전 불일치**로 인한 캐시/메타데이터 파싱 실패 |
