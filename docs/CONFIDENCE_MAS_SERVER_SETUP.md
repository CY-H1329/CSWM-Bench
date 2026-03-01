# Confidence MAS v2 — 5 Specialist LLM 서버 설정

5개 specialist (qwen3_4b, sa2va, llava4d, spatial_rgpt, spatial_reasoner)를 모두 사용하려면 아래 설정이 필요합니다.

## 1. SpatialRGPT

```bash
cd /home/jovyan/CY  # 또는 적절한 경로
git clone https://github.com/AnjieCheng/SpatialRGPT
export SPATIALRGPT_PATH=/home/jovyan/CY/SpatialRGPT
```

Jupyter에서 영구 적용:
```python
import os
os.environ["SPATIALRGPT_PATH"] = "/home/jovyan/CY/SpatialRGPT"
```

또는 `~/.bashrc` / `~/.zshrc`에 추가:
```bash
export SPATIALRGPT_PATH=/home/jovyan/CY/SpatialRGPT
```

## 2. Sa2VA (bitsandbytes/peft 충돌)

`TypeError: metaclass conflict` 발생 시:

```bash
# peft, bitsandbytes, transformers 버전 호환 확인
pip install peft==0.10.0  # 또는 Sa2VA 요구 버전
pip install bitsandbytes  # CUDA 버전에 맞게
```

Sa2VA가 특정 transformers/peft 버전을 요구할 수 있음.  
여전히 실패하면 `specialist_llms`로 Sa2VA 제외:

```python
results = run_confidence_mas_test(
    ...,
    specialist_llms=["qwen3_4b", "llava4d", "spatial_rgpt", "spatial_reasoner"],
)
```

## 3. 확인

```bash
# SpatialRGPT 경로 확인
echo $SPATIALRGPT_PATH
```

## 4. 요약

| Model | 요구사항 |
|-------|----------|
| qwen3_4b | transformers>=4.51 |
| llava4d | — |
| spatial_reasoner | — |
| spatial_rgpt | SPATIALRGPT_PATH 설정 |
| sa2va | peft/bitsandbytes 호환 버전 |
