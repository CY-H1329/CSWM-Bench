# SpatialRGPT + datasets 호환성 상세 분석

## 1. 모델 vs 데이터 로딩 — 혼동 금지

**SpatialRGPT 모델은 정상 로드됩니다.**

- **롤 프롬프트**: 가능
- **이미지 입력**: 가능 (VLM)
- **실제 오류**: **벤치마크 데이터셋 로딩 단계**에서 발생 (모델 추론 전)

---

## 2. 오류 발생 흐름

```
load_benchmark("cvbench")
  → load_from_disk (frozen) 실패 [datasets 버전 불일치]
  → load_dataset("nyu-visionx/CV-Bench") fallback
  → DatasetBuilder.__init__
  → DatasetInfo.from_directory(self._cache_dir)
  → Features.from_dict(self.features)
  → generate_from_dict() → fields(class_type)
  → TypeError: must be called with a dataclass type or instance
```

---

## 3. 근본 원인 (상세)

### 3.1 캐시 삭제해도 왜 계속 실패하나?

**HuggingFace에 올라간 CV-Bench 메타데이터 자체가 새 형식입니다.**

| 단계 | 설명 |
|------|------|
| 1 | `load_dataset("nyu-visionx/CV-Bench")` 호출 |
| 2 | datasets가 HF에서 다운로드 (parquet + dataset card 등) |
| 3 | `DatasetInfo.from_directory(cache_dir)` — **HF repo에 있는 metadata** 사용 |
| 4 | CV-Bench repo의 dataset card/parquet schema는 **최신 datasets**로 생성됨 |
| 5 | datasets 2.16.1이 이 형식을 파싱하지 못함 |

캐시를 지워도, **다시 받는 메타데이터가 같은 새 형식**이라 2.16.1로는 파싱이 불가능합니다.

### 3.2 Python 3.10과의 관계

Python 3.10은 이 오류와 **무관**합니다.  
문제는 `datasets` 버전과 HF에 저장된 메타데이터 형식의 불일치입니다.

### 3.3 datasets 2.16.1 vs 최신

- `Features.from_dict()` 내부에서 `generate_from_dict()` 호출
- 중첩 feature (`List[Value]`, `Image` 등) 처리 시 `dataclasses.fields(class_type)` 사용
- 최신 datasets는 feature 정의 구조가 바뀌어, 2.16.1의 `fields()` 호출 방식과 맞지 않음

---

## 4. 해결 방법

### 방법 A: parquet 직접 로드 (구현됨)

`load_dataset`이 실패하면 **parquet 로더**로 fallback합니다.  
parquet는 스키마를 파일에서 추론하므로 `dataset_info.json`을 사용하지 않습니다.

```python
# loaders.py에 이미 추가됨
data_files = {"test": [url_2d, url_3d]}
ds = load_dataset("parquet", data_files=data_files, split="test")
```

### 방법 B: datasets 업그레이드

```bash
pip install "datasets>=2.18"
```

vila와 의존성 충돌이 있을 수 있음.

### 방법 C: spatial_reasoning 환경 사용

Python 3.9 + 패치로 SpatialRGPT를 `spatial_reasoning`에서 실행.  
해당 환경의 datasets 버전이 HF 메타데이터와 호환됨.

---

## 5. 요약

| 질문 | 답변 |
|------|------|
| 캐시 삭제만으로 해결되나? | **아니요** — HF 메타데이터 자체가 새 형식 |
| Python 3.10 때문인가? | **아니요** — datasets 버전 문제 |
| 실제 원인은? | **datasets 2.16.1이 HF CV-Bench 메타데이터 형식을 파싱하지 못함** |
| 해결책은? | **parquet 직접 로드** (loaders.py에 fallback 추가됨) |
