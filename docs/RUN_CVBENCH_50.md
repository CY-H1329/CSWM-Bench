# CV-Bench 50샘플 테스트

## 서버에서 실행

```bash
# 1. Pull
cd Spatial_MAS  # 또는 프로젝트 경로
git pull origin main

# 2. 환경 확인 (timm, transformers, torch 등)
pip install timm transformers torch  # 필요시

# 3. CV-Bench 50샘플 실행 (25 train + 25 test)
python run_eval_mas_v2.py --benchmark cvbench --max_samples 50

# 4. 50개 전부 테스트만 (train 스킵, score map 랜덤)
python run_eval_mas_v2.py --benchmark cvbench --max_samples 50 --train_ratio 0
```

## 출력

- **Train phase**: 25샘플로 score map 업데이트
- **Test phase**: 25샘플로 accuracy 측정 (score map 고정)
- **결과**: `results/mas_v2/cvbench/<timestamp>/` 에 저장
  - `summary.json`: train/test accuracy, per_category
  - `test_details.jsonl`: 샘플별 상세

## Accuracy 확인

```bash
# summary.json에서 확인
cat results/mas_v2/cvbench/*/summary.json | jq '.test_accuracy, .test_per_category'
```

## 3DSRBench (나중에)

```bash
python run_eval_mas_v2.py --benchmark 3dsrbench --max_samples 50
```
