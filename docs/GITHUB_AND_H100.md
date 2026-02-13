# GitHub 푸시 & Pull & H100에서 실행

**저장소 주소:** https://github.com/CY-H1329/Spatial_MAS

---

## 1. 로컬에서 GitHub에 푸시 (최초 1회)

프로젝트 루트에서:

```bash
cd ~/Desktop/Spatial_MAS

# 저장소 초기화 (이미 되어 있으면 생략)
git init
git add .
git commit -m "Initial: STVQA-7K eval, failure analysis, export failed samples"

# 이 저장소로 연결
git remote add origin https://github.com/CY-H1329/Spatial_MAS.git
# SSH 사용 시: git remote add origin git@github.com:CY-H1329/Spatial_MAS.git

git branch -M main
git push -u origin main
```

- `results/` 는 `.gitignore`에 있으므로 푸시되지 않습니다.
- 이후 변경 후: `git add . && git commit -m "메시지" && git push`

---

## 2. Pull — GitHub에서 최신 코드 받기

다른 PC(H100 등)에서 이미 clone 해 둔 폴더를 **최신 상태로 맞출 때**, 또는 로컬에서 GitHub에 올린 뒤 다른 곳에서 받을 때:

```bash
cd /path/to/Spatial_MAS
git pull origin main
```

- **처음 이 PC에서 받을 때**는 clone 한 번만 하면 됨 (아래 3단계).
- **이미 clone 한 폴더**가 있으면, 그 폴더 안에서 위처럼 `git pull` 만 하면 됨.

---

## 3. H100 서버에서 처음 받기 (clone) & 환경 설정

```bash
# 저장소 클론 (최초 1회)
git clone https://github.com/CY-H1329/Spatial_MAS.git
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

## 4. H100에서 실행 순서

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

## 5. 이후 코드 반영 (로컬 ↔ H100)

- **로컬에서 수정 후 GitHub에 푸시**
  ```bash
  cd ~/Desktop/Spatial_MAS
  git add .
  git commit -m "설명"
  git push origin main
  ```

- **H100(또는 다른 PC)에서 최신 받기 (Pull)**
  ```bash
  cd /path/to/Spatial_MAS
  git pull origin main
  conda activate spatial_mas
  # 필요 시: pip install -r requirements.txt
  ```

이후 위 "3. H100에서 실행 순서"(평가 → 분석 → 틀린 샘플 저장)대로 다시 실행하면 됩니다.

---

## 6. Push / Pull — MAS 모델 (Qwen3-4B, Sa2VA, LLaVA4D) 추가 후

### 로컬 → GitHub (Push)

```bash
cd ~/Desktop/Spatial_MAS
git add .
git status                    # 확인
git commit -m "Add Qwen3-4B, Sa2VA, LLaVA4D runners; MAS pipeline"
git push origin main
```

### H100 서버 → GitHub (Pull)

```bash
cd /path/to/Spatial_MAS
git pull origin main
conda activate spatial_mas
pip install -r requirements.txt   # transformers>=4.51, etc.
```

### H100에서 MAS 실행

```bash
# 1) Datasets (최초 1회)
python scripts/setup_datasets.py

# 2) MAS 평가 (Qwen3-4B × 3 agents)
bash scripts/run_h100_mas.sh stvqa7k qwen3_4b qwen3_4b qwen3_4b --max_per_category 10

# 3) 다른 조합
bash scripts/run_h100_mas.sh stvqa7k sa2va sa2va sa2va
bash scripts/run_h100_mas.sh stvqa7k qwen3_4b llava4d sa2va
```
