# Git & Server Setup

Repository: https://github.com/CY-H1329/Spatial_MAS

---

## 1. Clone (first time)

```bash
git clone https://github.com/CY-H1329/Spatial_MAS.git
cd Spatial_MAS
```

## 2. Environment setup (first time)

```bash
conda env create -f environment.yml
conda activate spatial_mas
python scripts/setup_datasets.py
```

**GPU (H100, CUDA 12.x):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Push / Pull workflow

| Action | Command |
|--------|---------|
| **Pull** (get latest) | `cd Spatial_MAS && git pull origin main` |
| **Push** (after edits) | `git add . && git commit -m "message" && git push origin main` |

**Paths:**
- Local: `~/Desktop/Spatial_MAS`
- Server (H100): `~/CY/Spatial_MAS`

## 4. Push results from H100

```bash
cd ~/CY/Spatial_MAS
python scripts/gather_results_summary.py
git add results_summary/
git commit -m "Results: 3DSRBench summaries"
git push origin main
```

See [H100_PUSH_RESULTS.md](H100_PUSH_RESULTS.md) for details.

## 5. Authentication

GitHub no longer accepts password. Use:
- **Personal Access Token** (Settings → Developer settings → Personal access tokens)
- Or **SSH**: `git remote set-url origin git@github.com:CY-H1329/Spatial_MAS.git`
