# TypeError: must be called with a dataclass type or instance — 상세 분석

## 1. 오류 요약

```
TypeError: must be called with a dataclass type or instance
```

- **발생 위치**: `datasets` 라이브러리 내부 `Features.from_dict()` → `generate_from_dict()` → `dataclasses.fields()`
- **발생 시점**: CV-Bench (또는 유사 HF 데이터셋) 로딩 시
- **Python 3.10과의 관계**: **무관** — Python 버전이 아니라 `datasets` 버전 문제

---

## 2. 스택 트레이스 해석 (단계별)

```
load_benchmark("cvbench", ...)
  ↓
load_from_disk(frozen_path)  ← 실패 (dataclass/datasets 버전 불일치)
  ↓ [경고 후 fallback]
load_dataset("nyu-visionx/CV-Bench", split="test")
  ↓
load_dataset_builder(...)
  ↓
DatasetBuilder.__init__
  ↓
DatasetInfo.from_directory(self._cache_dir)   ← HF 캐시의 dataset_info.json 사용
  ↓
DatasetInfo.from_dict(dataset_info_dict)
  ↓
Features.from_dict(self.features)             ← 여기서 features 구조 파싱
  ↓
generate_from_dict(dic)
  ↓ (중첩 feature 처리)
generate_from_dict(value) for key, value in obj.items()
  ↓
field_names = {f.name for f in fields(class_type)}   ← dataclasses.fields() 호출
  ↓
TypeError: must be called with a dataclass type or instance
```

### 2.1 `dataclasses.fields()`가 실패하는 이유

`dataclasses.fields(obj)`는 **dataclass 타입 또는 인스턴스**만 받습니다.

- `datasets` 2.16.1의 `generate_from_dict()`는 feature 정의를 파싱할 때,  
  **최신 datasets에서 사용하는 중첩 구조**를 `fields()`에 넘기려 함
- 최신 datasets는 feature 스키마 형식이 바뀌어,  
  2.16.1이 기대하는 "dataclass 타입/인스턴스" 형태가 아님
- 그 결과 `fields()`에 dict, list, 또는 다른 타입이 전달되어 `TypeError` 발생

---

## 3. 왜 캐시 삭제해도 해결되지 않는가?

| 단계 | 설명 |
|------|------|
| 1 | `load_dataset("nyu-visionx/CV-Bench")` 호출 |
| 2 | datasets가 HuggingFace에서 다운로드 (parquet, dataset card, metadata 등) |
| 3 | HF repo에 저장된 `dataset_info.json` / metadata 사용 |
| 4 | **CV-Bench repo의 메타데이터는 최신 datasets로 생성됨** |
| 5 | datasets 2.16.1이 이 형식을 파싱하지 못함 |

캐시를 지워도, **다시 받는 메타데이터가 동일한 새 형식**이므로 2.16.1로는 파싱이 불가능합니다.

---

## 4. srgpt 환경의 제약

- **datasets==2.16.1**: SpatialRGPT/vila 의존성으로 고정
- **pip install datasets>=2.18**: 로딩은 해결되지만 vila와 충돌 가능

---

## 5. 해결 방법 (우선순위)

### 5.1 parquet 직접 로드 (권장, 이미 구현됨)

`loaders.py`에 **parquet fallback**이 추가되어 있습니다.

- `load_dataset("nyu-visionx/CV-Bench")`가 dataclass 관련 오류로 실패하면
- `load_dataset("parquet", data_files={...}, split="test")`로 전환
- parquet 로더는 **파일에서 스키마를 추론**하므로 `dataset_info.json`을 사용하지 않음

**서버 코드가 최신인지 확인하세요.**  
로컬(`Spatial_MAS-main`)과 서버(`/home/jovyan/CY/Spatial_MAS`)가 다를 수 있습니다.

```bash
# 서버에서 loaders.py 확인
grep -A5 "parquet fallback" /home/jovyan/CY/Spatial_MAS/src2/benchmarks/loaders.py
```

### 5.2 datasets 업그레이드

```bash
pip install "datasets>=2.18"
```

vila와의 호환성은 별도 확인 필요.

### 5.3 spatial_reasoning 환경 사용

Python 3.9 + 패치로 SpatialRGPT를 `spatial_reasoning` 환경에서 실행.  
해당 환경의 datasets 버전이 HF 메타데이터와 호환됩니다.

---

## 6. 3DSRBench에서 동일 오류가 발생한다면

3DSRBench(`ccvl/3DSRBench`)도 같은 방식으로 메타데이터가 생성되었다면  
동일한 `TypeError`가 발생할 수 있습니다.  
이 경우 `loaders.py`에 3DSRBench용 parquet(또는 다른 형식) fallback을 추가해야 합니다.

---

## 7. 요약 표

| 항목 | 내용 |
|------|------|
| **오류 원인** | datasets 2.16.1이 HF CV-Bench 메타데이터(최신 형식)를 파싱하지 못함 |
| **Python 3.10** | 관련 없음 |
| **캐시 삭제** | HF 메타데이터가 새 형식이므로 효과 없음 |
| **해결책** | parquet 직접 로드 fallback (loaders.py에 구현됨) |
| **확인 사항** | 서버의 `loaders.py`가 최신 버전인지 확인 |
