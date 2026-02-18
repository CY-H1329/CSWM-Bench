# Head-Agent 결과 Push/Pull 및 표 정리

## 1. H100 서버에서 (실험 후)

### 1.1 결과 수집
```bash
cd ~/CY/Spatial_MAS
conda activate spatial_mas  # 또는 spatialeval_orchestration

python scripts/gather_results_summary.py
```
→ `results/runs/head_agent/` 의 CV-Bench, 3DSRBench category routing 결과가 `results_summary/head_agent/` 로 복사됨

### 1.2 Push
```bash
git add results_summary/
git status
git commit -m "Results: Head-Agent CV-Bench + 3DSRBench"
git push origin main
```

---

## 2. 로컬에서

### 2.1 Pull
```bash
cd ~/Desktop/Spatial_MAS
git pull origin main
```

### 2.2 표 정리
```bash
python scripts/summarize_head_agent_results.py --output results_summary/HEAD_AGENT_SUMMARY.md
```
→ `results_summary/HEAD_AGENT_SUMMARY.md` 에 표가 생성됨

### 2.3 (선택) Push 정리 결과
```bash
git add results_summary/HEAD_AGENT_SUMMARY.md
git commit -m "Head-Agent summary table"
git push origin main
```

---

## 3. H100 서버에서 (정리된 표 확인 또는 직접 생성)

```bash
cd ~/CY/Spatial_MAS
git pull origin main
cat results_summary/HEAD_AGENT_SUMMARY.md
```

**또는** 로컬 pull 전에 H100에서 직접 표 생성:
```bash
python scripts/summarize_head_agent_results.py --dir results --output results_summary/HEAD_AGENT_SUMMARY.md
git add results_summary/HEAD_AGENT_SUMMARY.md
git commit -m "Head-Agent summary table"
git push origin main
```

---

## 요약

| 단계 | 위치 | 명령 |
|------|------|------|
| 1. 수집 | H100 | `python scripts/gather_results_summary.py` |
| 2. Push | H100 | `git add results_summary/ && git commit -m "..." && git push` |
| 3. Pull | 로컬 | `git pull origin main` |
| 4. 표 생성 | 로컬 | `python scripts/summarize_head_agent_results.py --output results_summary/HEAD_AGENT_SUMMARY.md` |
| 5. (선택) Push | 로컬 | `git add ... && git push` |
| 6. 확인 | H100 | `git pull && cat results_summary/HEAD_AGENT_SUMMARY.md` |
