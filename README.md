# Spatial_MAS

STVQA-7K 데이터셋으로 **Qwen2.5-VL**, **LLaVA**, **GPT** 모델의 공간 추론 성능을 평가하는 프로젝트입니다.

- **데이터셋**: [STVQA-7K](https://huggingface.co/datasets/OX-PIXL/STVQA-7K) (SpatialThinker, arXiv:2511.07403)
- **목적**: 학습용 데이터셋을 **평가 벤치마크**로 사용해 위 모델들의 성능 비교

---

## 1. 환경 구성 (Conda)

```bash
cd ~/Desktop/Spatial_MAS
conda env create -f environment.yml
conda activate spatial_mas
```

서버 CUDA 버전에 맞게 `environment.yml`의 `cudatoolkit`을 수정할 수 있습니다 (예: H100은 CUDA 12.x).

---

## 2. 설정

- **config.yaml**: 데이터셋 스플릿(`val`/`train`), 모델 ID, GPT API 키 등.
- **GPT 사용 시**: 환경변수 `OPENAI_API_KEY` 설정.

```bash
export OPENAI_API_KEY=sk-...
```

---

## 3. 실행

### 전체 평가 (Qwen + LLaVA + GPT)

```bash
python run_eval.py --models qwen llava gpt --split val
```

### 특정 모델만

```bash
python run_eval.py --models qwen --split val
python run_eval.py --models gpt --split val --max_samples 100
```

### 디버깅 (소량 샘플)

```bash
python run_eval.py --models qwen llava --split val --max_samples 50
```

결과는 `results/YYYYMMDD_HHMMSS/` 아래에 저장됩니다.

- `*_results.json`: 정확도, (옵션) 카테고리별 정확도
- `*_preds.jsonl`: 샘플별 예측 (config에서 `save_predictions: true`일 때)
- `summary.json`: 모델별 요약

### 실패 질문을 task(category)별로 분석

어떤 **task 유형**에서 틀리는지 보려면, 평가 후 다음을 실행하세요.

```bash
python analyze_failures.py --run_dir results/20250109_123456
```

- `--run_dir`: 방금 돌린 run 디렉터리 (날짜/시간 폴더)
- `--models qwen gpt`: 특정 모델만 분석
- `--output report.md`: 보고서를 한 파일로 저장
- `--no_verbose`: 요약 통계만, 실패 질문 목록은 생략

출력: task(category)별 **총 개수 / 정답 수 / 실패 수 / 정확도(%)** 테이블과, 실패한 질문 목록(최대 20개씩).  
추가로 `failure_analysis_<model>.json`이 run_dir에 저장됩니다.

### 틀린 샘플만 따로 저장 (이미지 + 질문/답변 등 전체 정보)

틀린 데이터만 모아서 **이미지 + 질문/옵션/정답/예측/category** 전부 저장해 두고, 그 안에서 task별로 정리하려면:

```bash
python export_failed_samples.py --run_dir results/20250109_123456
```

- **저장 위치**: `run_dir/failed_samples/<model>/`
  - `failed_manifest.jsonl`: 한 줄에 한 틀린 샘플 (JSON). `idx`, `category`, `question_only`, `options`, `answer_gt`, `answer_pred`, `answer_text_gt`, `answer_text_pred`, `level`, `rating`, `image_path`, `image_id` 등
  - `by_category/<category>/`: task별 폴더, 그 안에 `img_<idx>.png` 이미지
  - `failed_summary.json`, `README.md`
- `--models qwen gpt`: 특정 모델만 내보내기  
- `--out_dir /other/path`: 저장 경로 변경

---

## 4. GitHub 푸시 & H100에서 Pull 후 실행

- **로컬에서 GitHub에 푸시**: [docs/GITHUB_AND_H100.md](docs/GITHUB_AND_H100.md) 참고.
- **H100에서**: Pull 후 **환경 설정 1회** → **실행 스크립트** 한 번에 끝.
  ```bash
  cd ~/CY/Spatial_MAS
  git pull origin main
  bash scripts/setup_h100.sh    # 최초 1회: conda env, API 키 안내
  echo 'export OPENAI_API_KEY=sk-키' > .env   # GPT용, 1회
  bash scripts/run_h100.sh     # 평가 + 실패 분석 + 틀린 샘플 저장
  ```
  자세한 순서·API 키 설정: **[docs/H100_설정_및_실행.md](docs/H100_설정_및_실행.md)**.

---

## 5. JupyterHub H100 서버에서 실행

1. **프로젝트 업로드**  
   `Spatial_MAS` 폴더를 서버 홈 또는 작업 디렉터리로 업로드.

2. **Conda 환경 생성** (서버 터미널 또는 노트북 셀)

   ```bash
   cd /path/to/Spatial_MAS
   conda env create -f environment.yml
   conda activate spatial_mas
   ```

3. **커널 등록** (Jupyter에서 이 env 사용 시)

   ```bash
   python -m ipykernel install --user --name spatial_mas --display-name "spatial_mas"
   ```

4. **실행**
   - 터미널: `python run_eval.py --models qwen llava gpt --split val`
   - 또는 `notebooks/run_eval.ipynb` 사용 (같은 명령을 노트북에서 실행).

5. **H100 관련**
   - `config.yaml`의 `device: "cuda"` 그대로 두면 기본 GPU 사용.
   - 여러 GPU가 있으면 `CUDA_VISIBLE_DEVICES=0 python run_eval.py ...` 로 지정 가능.

---

## 6. 프로젝트 구조

```
Spatial_MAS/
├── environment.yml           # Conda 환경
├── config.yaml               # 데이터/모델/출력 설정
├── .env.example              # API 키 예시 (복사해 .env 로 사용)
├── run_eval.py               # 평가 실행
├── analyze_failures.py       # 실패 질문 task별 분석
├── export_failed_samples.py  # 틀린 샘플만 이미지+전체 정보 저장
├── scripts/
│   ├── setup_h100.sh         # H100 환경 설정 (최초 1회)
│   └── run_h100.sh           # H100 평가+분석+export 한 번에 실행
├── README.md
├── docs/
│   ├── GITHUB_AND_H100.md    # GitHub 푸시 & H100 실행 방법
│   └── H100_설정_및_실행.md  # H100 Pull 후 설정·실행 순서
├── src/
│   ├── data.py
│   └── models/ (qwen, llava, gpt)
├── notebooks/
│   └── run_eval.ipynb
└── results/                  # run별 결과 (results/날짜시간/failed_samples/ 등)
```

---

## 7. 참고

- 데이터셋: [OX-PIXL/STVQA-7K](https://huggingface.co/datasets/OX-PIXL/STVQA-7K) (또는 `hunarbatra/STVQA-7K`)
- 논문: [SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](https://arxiv.org/abs/2511.07403)
