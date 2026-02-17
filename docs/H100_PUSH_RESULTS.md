# H100: Push Results to GitHub

Steps to push experiment results from H100 to GitHub so they can be pulled and organized locally.

## 1. Gather results into results_summary/

```bash
cd ~/CY/Spatial_MAS
conda activate spatial_mas

python scripts/gather_results_summary.py
```

This copies key files from `results/` to `results_summary/`:
- `category_*.csv`, `category_*.json`, `summary.txt` (API)
- `results.json` (GPU per run)

## 2. Commit and push

```bash
git add results_summary/
git status
git commit -m "Results: 3DSRBench API + GPU summaries"
git push origin main
```

(Use Personal Access Token as password if prompted.)

## 3. On local: Pull and organize

```bash
cd ~/Desktop/Spatial_MAS
git pull origin main
```

The `results_summary/` folder is now updated. You can review and further organize for the paper.
