# eval_h100 — H100 서버에서 실행

Spatial_MAS agent runners 및 평가를 H100 서버에서 실행하기 위한 스크립트 모음.

## 1. GitHub에서 Pull

```bash
# H100 서버에서
cd ~/  # 또는 원하는 경로
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS
```

이미 clone 되어 있다면:

```bash
cd Spatial_MAS
git pull origin main
```

## 2. 설정 (최초 1회)

```bash
cd Spatial_MAS
bash eval_h100/setup.sh
```

- Conda env `spatial_mas` 생성
- pip 의존성 설치
- CUDA 확인

## 3. 간단 테스트 (모델 설치 확인)

```bash
cd Spatial_MAS
bash eval_h100/test_runners.sh
```

또는:

```bash
conda activate spatial_mas
python eval_h100/test_runners.py --model qwen3_4b
```

- `--model qwen3_4b` : Qwen3-VL-4B (기본, 권장)
- `--model sa2va` : Sa2VA
- `--model llava4d` : LLaVA-1.6 (proxy)
- `--skip-inference` : import만 확인, 모델 다운로드 없음

## 4. 평가 실행

```bash
bash eval_h100/run_eval.sh
# 또는 100 샘플:
bash eval_h100/run_eval.sh 100
```

## 5. GitHub에 Push (로컬에서 수정 후)

```bash
cd Spatial_MAS
git add eval_h100/
git add src/models/   # runner 변경 시
git commit -m "Add eval_h100 for H100 deployment"
git push origin main
```

## 요구사항

- Python 3.10, Conda
- CUDA (H100)
- HuggingFace 로그인 (모델 다운로드): `huggingface-cli login`

## 선택 사항

- **SpatialRGPT**: `export SPATIALRGPT_PATH=/path/to/SpatialRGPT` (repo clone 필요)
- **API 모델** (GPT, Claude, Gemini): `.env`에 API 키 설정
