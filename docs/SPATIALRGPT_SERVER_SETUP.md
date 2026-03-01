# SpatialRGPT 서버 실행 가이드

Spatial_MAS에서 SpatialRGPT를 서버에서 실행하기 위한 설정 방법입니다.

## 요구사항

- Python 3.9+ (spatial_reasoning env)
- CUDA GPU (~16GB+ VRAM 권장)
- `accelerate` 없이 동작 (device_map 미사용)

## 1. SpatialRGPT repo 클론

```bash
cd ~/CY  # 또는 원하는 경로
git clone https://github.com/AnjieCheng/SpatialRGPT
```

## 2. SpatialRGPT 패치 적용 (device_map 제거)

Spatial_MAS와 함께 사용하려면 SpatialRGPT의 `llava/model/builder.py`를 수정해야 합니다.

**방법 A**: 로컬에서 수정된 SpatialRGPT 폴더를 서버로 복사 (scp, rsync 등)

**방법 B**: 서버에서 직접 수정

`SpatialRGPT/llava/model/builder.py`:

1. **prepare_config_for_eval** (약 228행): `device_map`이 None이면 덮어쓰지 않음
```python
# before
if "siglip" in vision_tower_name.lower():
    kwargs["device_map"] = "cuda"
# after
if "siglip" in vision_tower_name.lower() and kwargs.get("device_map") is not None:
    kwargs["device_map"] = "cuda"
```

2. **load_pretrained_model** (약 48행): `device != "cuda"`일 때만 device_map 덮어쓰기
```python
# before
if device != "cuda":
    kwargs["device_map"] = {"": device}
# after
if device != "cuda" and device_map is not None:
    kwargs["device_map"] = {"": device}
```

3. **model.eval().cuda()** (약 182행): `device_map=None`이면 `model.to(device)` 사용
```python
# before
model.eval().cuda()
# after
model.eval()
if device_map is None:
    model.to(device)
```

## 3. 환경 변수 설정

```bash
export SPATIALRGPT_PATH=/path/to/SpatialRGPT
# 예: export SPATIALRGPT_PATH=/home/jovyan/CY/SpatialRGPT
```

`.bashrc` 또는 `.zshrc`에 추가하면 영구 적용됩니다.

## 4. SpatialRGPT 의존성 (선택)

Spatial_MAS의 `spatial_reasoning` env에서 실행할 경우, SpatialRGPT를 **pip install 하지 않고** `SPATIALRGPT_PATH`만 설정하면 됩니다.

필요 시 SpatialRGPT의 의존성을 설치할 수 있습니다 (Python 3.10 권장):

```bash
cd SpatialRGPT
./environment_setup.sh srgpt
conda activate srgpt
```

단, `spatial_reasoning` env와 충돌할 수 있으므로, **Spatial_MAS 테스트만 할 경우**는 `SPATIALRGPT_PATH`만 설정하고 `spatial_reasoning` env를 사용하는 것이 좋습니다.

## 5. 실행

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_reasoning
export SPATIALRGPT_PATH=/path/to/SpatialRGPT

python test_specialist_all_roles.py --model spatial_rgpt --max_samples 10
```

## 6. 모델

- 기본: `a8cheng/SpatialRGPT-VILA1.5-8B` (8B, HuggingFace에서 자동 다운로드)
- VRAM 부족 시: 더 작은 체크포인트 사용 가능

## 7. 문제 해결

| 문제 | 해결 |
|------|------|
| `SPATIALRGPT_PATH` not set | `export SPATIALRGPT_PATH=/path/to/SpatialRGPT` |
| `accelerate` required | SpatialRGPT `builder.py` 패치 적용 확인 |
| `device_map` error | `device_map=None`으로 `load_pretrained_model` 호출 확인 |
| Python 3.9 호환 | `builder.py` 패치가 Python 3.9에서 동작하도록 수정됨 |

## 8. 관련 파일

| 파일 | 역할 |
|------|------|
| `src2/models/spatial_rgpt.py` | SpatialRGPTRunner (device_map=None) |
| `SpatialRGPT/llava/model/builder.py` | load_pretrained_model (패치 필요) |
