# GitHub 푸시 & H100에서 실행

## 1. 로컬에서 GitHub에 푸시 (최초 1회)

프로젝트 루트에서:

```bash
cd ~/Desktop/Spatial_MAS

# 저장소 초기화 (이미 되어 있으면 생략)
git init
git add .
git commit -m "Initial: STVQA-7K eval, failure analysis, export failed samples"

# GitHub에서 새 저장소 생성 후 (예: YOUR_USERNAME/Spatial_MAS)
git remote add origin https://github.com/YOUR_USERNAME/Spatial_MAS.git
# 또는 SSH: git remote add origin git@github.com:YOUR_USERNAME/Spatial_MAS.git

git branch -M main
git push -u origin main
```

- `results/` 는 `.gitignore`에 있으므로 푸시되지 않습니다.
- 이후 변경 후: `git add . && git commit -m "메시지" && git push`

---

## 2. H100 서버에서 clone & 환경 설정

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/Spatial_MAS.git
cd Spatial_MAS

# Conda 환경 생성
conda env create -f environment.yml
conda activate spatial_mas
```

**CUDA용 PyTorch (H100 등 GPU 서버):**  
환경에 이미 포함된 PyTorch가 GPU를 못 쓰면:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Jupyter 커널 등록 (선택):**

```bash
python -m ipykernel install --user --name spatial_mas --display-name "spatial_mas"
```

---

## 3. H100에서 실행 순서

### 1) 평가 실행

```bash
cd /path/to/Spatial_MAS
conda activate spatial_mas

# 필요 시 GPU 지정
export CUDA_VISIBLE_DEVICES=0

# Qwen + LLaVA (GPU), GPT는 API 키 필요
export OPENAI_API_KEY=sk-...

python run_eval.py --models qwen llava gpt --split val
# 빠른 테스트: --max_samples 100
```

결과는 `results/YYYYMMDD_HHMMSS/` 에 생성됩니다.

### 2) 실패 질문 task별 분석

```bash
python analyze_failures.py --run_dir results/YYYYMMDD_HHMMSS
# 특정 모델만: --models qwen gpt
# 보고서 파일로: --output results/YYYYMMDD_HHMMSS/report.md
```

### 3) 틀린 샘플만 따로 저장 (이미지 + 질문/답변 등 전체 정보)

```bash
python export_failed_samples.py --run_dir results/YYYYMMDD_HHMMSS
```

생성 구조:

```
results/YYYYMMDD_HHMMSS/failed_samples/
  qwen/
    failed_manifest.jsonl      # 한 줄에 한 틀린 샘플 (전체 메타)
    failed_summary.json         # category별 실패 개수
    README.md
    by_category/
      relation/                # task별 폴더
        img_00012.png
        img_00045.png
      depth/
        ...
  gpt/
    ...
```

- **failed_manifest.jsonl**: 각 줄이 JSON. `idx`, `category`, `question_only`, `options`, `answer_gt`, `answer_pred`, `answer_text_gt`, `answer_text_pred`, `level`, `rating`, `image_path`, `image_id` 등 포함.
- **by_category/** 에서 task별로 이미지 확인 후, 필요하면 그 안에서 다시 정리하면 됩니다.

---

## 4. 이후 코드 반영 (로컬 ↔ H100)

- **로컬에서 수정 후 GitHub에 푸시**
  ```bash
  git add .
  git commit -m "설명"
  git push
  ```

- **H100에서 최신 받기**
  ```bash
  cd /path/to/Spatial_MAS
  git pull
  conda activate spatial_mas
  # 필요 시: pip install -r requirements.txt
  ```

이후 위 3단계(평가 → 분석 → 틀린 샘플 저장) 순서대로 다시 실행하면 됩니다.
