# MAS v2 지연 시간 분석

## 샘플당 실행 흐름

```
1. Head Agent          → 1x VLM (Qwen3-VL-4B)
2. direct_visual       → 1x VLM (no tools)
3. explicit_3d         → object_extraction (1x VLM) + get_3d_representation + 1x VLM
4. scene_graph         → object_extraction (1x VLM) + get_scene_graph + 1x VLM
5. Final Reasoning     → 1x DeepSeek-R1-Distill (text-only)
```

---

## 시간 소요 구성요소

| 단계 | 작업 | 모델/도구 | 대략적 시간 |
|------|------|-----------|-------------|
| 1 | Head | Qwen3-VL-4B | ~5–15초 |
| 2 | direct_visual | Qwen3-VL-4B | ~10–30초 |
| 3a | object_extraction | Qwen3-VL-4B | ~10–30초 |
| 3b | get_depth_map | DepthAnything | ~3–10초 |
| 3c | get_detections (OWL-ViT) | OWL-ViT | ~2–5초 |
| 3d | explicit_3d specialist | Qwen3-VL-4B | ~15–40초 |
| 4a | object_extraction | Qwen3-VL-4B | **중복** |
| 4b | get_scene_graph (OWL-ViT) | OWL-ViT | **중복** |
| 4c | scene_graph specialist | Qwen3-VL-4B | ~15–40초 |
| 5 | Final Reasoning | DeepSeek-R1-Distill | ~20–60초 |

**총 VLM 추론**: 7회 (Head 1 + direct_visual 1 + object_extraction 2 + specialist 3)  
**총 OWL-ViT**: 2회 (explicit_3d, scene_graph 각 1회)  
**DepthAnything**: 1회

---

## 주요 병목

### 1. VLM 추론 7회 (가장 큰 비중)

- Qwen3-VL-4B: 이미지 + 긴 프롬프트 → 1회당 10–40초
- 7회 합치면 대략 70–280초 (1–5분) 수준

### 2. object_extraction 2회 중복

- `explicit_3d`와 `scene_graph`가 각각 `extract_objects_from_image` 호출
- 같은 이미지, 같은 프롬프트 → 결과 동일
- 한 번만 호출하고 공유하면 VLM 1회 절약 (~10–30초)

### 3. OWL-ViT 2회 중복

- `get_3d_representation`과 `get_scene_graph`가 둘 다 `get_detections_with_labels` 사용
- 같은 이미지, 같은 `object_names` → 결과 동일
- object_names를 공유하면 OWL-ViT 1회 절약 (~2–5초)

### 4. DepthAnything 1회

- `get_depth_map`: monocular depth 추정
- 1회당 수 초 수준

### 5. Final Reasoning 1회

- DeepSeek-R1-Distill: 긴 텍스트 생성
- 1회당 20–60초 수준

---

## 개선 방향

### A. object_extraction 공유 (즉시 적용 가능)

- `explicit_3d`와 `scene_graph`가 같은 `object_names`를 사용하므로, 한 번만 추출 후 공유
- 예상 절감: VLM 1회 (~10–30초)

### B. OWL-ViT 결과 공유

- object_names 공유 시, detection 결과도 한 번만 계산 후 공유
- 예상 절감: OWL-ViT 1회 (~2–5초)

### C. Specialist 병렬 실행 (구조 변경)

- direct_visual, explicit_3d, scene_graph를 병렬로 실행
- GPU 메모리 여유가 있으면 3개 동시 실행으로 전체 wall-clock 시간 단축 가능

### D. object_extraction + tool 호출 선실행

- Head 이후, specialist 루프 전에 object_extraction과 3D/scene_graph 도구를 한 번에 실행
- 이후 specialist들은 캐시된 결과만 사용
