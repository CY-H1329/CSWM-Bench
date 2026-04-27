## Image-to-Video (I2V) quick eval on CSWM scenes

Goal: run a few **open** image→video models on two simple CSWM scenes:
- Door opens with/without obstacle
- Cup push small vs large (threshold)

This is for **fast qualitative evidence** (videos/GIFs) and a minimal
"contrastive divergence" score (optional).

### Recommended models (open, practical)

- **Stable Video Diffusion (SVD / SVD-XT)** via `diffusers` (image→video).
  - easiest to get running on H100
  - good for demo; not guaranteed physical correctness (that’s the point)

### Install (H100)

Use a clean env if possible:

```bash
conda create -n i2v python=3.10 -y
conda activate i2v
pip install -U pip
pip install -r scripts/evals/i2v_models/requirements-i2v.txt
```

### Generate CSWM images (20 samples) + run SVD

```bash
cd ~/CY/CSWM-Bench
python scripts/evals/cswmbench_api/generate_cswmbench.py

conda activate i2v
python scripts/evals/i2v_models/run_svd_on_cswm.py --max_items 20 --fps 8
python scripts/evals/i2v_models/make_video_viewer.py --run_dir results/runs/i2v/svd/latest
```

Outputs:
- `results/runs/i2v/svd/<timestamp>/videos/*.mp4`
- `results/runs/i2v/svd/<timestamp>/viewer/index.html`

Open the viewer and you’ll have slide-ready comparisons.

