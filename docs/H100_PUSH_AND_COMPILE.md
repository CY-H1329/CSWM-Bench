# H100: Push Results → 로컬 Pull → 표 정리

## 1. H100 서버에서 실행

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_mas   # 또는 spatialeval_orchestration

# 1) 결과 수집 + 표 정리 (CV-Bench, 3DSRBench category별)
python scripts/gather_results_summary.py
python scripts/compile_results_tables.py

# 2) Push
git add results_summary/
git status
git commit -m "Results: CV-Bench + 3DSRBench tables"
git push origin main
```

## 2. 로컬에서 Pull

```bash
cd ~/Desktop/Spatial_MAS
git pull origin main
```

## 3. 결과 확인

- `results_summary/SUMMARY_TABLES.md` — CV-Bench, 3DSRBench 표 (Overall + Category별)
- `results_summary/cvbench_by_category.csv` — CV-Bench 상세
- `results_summary/3dsrbench/*/category_performance.csv` — 3DSRBench 상세

## 요약

| 단계 | 위치 | 명령 |
|------|------|------|
| 1. 수집+정리 | H100 | `python scripts/gather_results_summary.py` + `python scripts/compile_results_tables.py` |
| 2. Push | H100 | `git add results_summary/ && git commit -m "..." && git push origin main` |
| 3. Pull | 로컬 | `git pull origin main` |
| 4. 확인 | 로컬 | `results_summary/SUMMARY_TABLES.md` |
