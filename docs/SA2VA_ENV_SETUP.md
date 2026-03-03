# Sa2VA + Spatial_MAS 가상환경 설정

Sa2VA(ByteDance/Sa2VA-4B)와 Qwen3-VL을 함께 사용하려면 호환되는 패키지 버전이 필요합니다.

## 1. 권장: transformers 4.57.1

Sa2VA 공식: `transformers==4.57.1` (또는 4.49.0 legacy)  
Qwen3-VL: `transformers>=4.51`  
→ **4.57.1** 사용 시 둘 다 만족.

## 2. 한 번에 설치

```bash
# 새 conda 환경
conda create -n spatial_mas python=3.10 -y
conda activate spatial_mas

# Spatial_MAS 프로젝트로 이동
cd ~/CY/Spatial_MAS  # 또는 실제 경로

# Sa2VA 호환 의존성 (한 번에)
pip install -r requirements-sa2va.txt
```

## 3. requirements-sa2va.txt 내용

```
torch>=2.1.0
torchvision
transformers==4.57.1
datasets
huggingface_hub
accelerate
peft
qwen-vl-utils
pillow
einops
timm
...
```

## 4. transformers 5.x 사용 시 (패치 적용됨)

이미 `transformers>=5.0`을 설치한 경우, `sa2va.py`에 `all_tied_weights_keys` 호환 패치가 포함되어 있습니다.  
그래도 오류가 나면 `transformers==4.57.1`로 다운그레이드하세요:

```bash
pip install transformers==4.57.1
```

## 5. 확인

```bash
python -c "
from src2.models.sa2va import Sa2VARunner
r = Sa2VARunner(device='cpu')  # CPU로 빠른 확인
print('Sa2VA OK')
"
```

## 6. 실행

```bash
python test_fixed_specialist_mas_v2.py --specialist sa2va --benchmark cvbench --max_samples 100 --category_filter Count
```
