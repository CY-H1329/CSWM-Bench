## CSWM-DMControl: contrastive divergence test (World Model evaluation scaffold)

This module builds **CSWM-style contrastive pairs** on a real simulator (DMControl):

- same env + same initial state
- two action sequences `A` and `A'` that differ in **one step** only (small delta)
- measure whether futures diverge (GT) and export rollouts as MP4

This is the cleanest “WM + spatial reasoning” PoC because:
- you have a simulator GT (no debate about correctness)
- you can demonstrate **absolute evaluation trap**: many models generate plausible futures but do not separate counterfactuals.

### Install (H100)

Create a fresh env:

```bash
conda create -n cswm_dm python=3.10 -y
conda activate cswm_dm
pip install -U pip
pip install dm_control numpy pillow tqdm imageio imageio-ffmpeg
```

If `dm_control` complains about MuJoCo, ensure `mujoco` is installed (pip will pull it) and that you have EGL/headless support in the container.

### Generate a small dataset (20 pairs) + videos

```bash
cd ~/CY/CSWM-Bench
python scripts/evals/cswm_dmcontrol/generate_pairs.py --env cartpole:swingup --n_pairs 20 --horizon 40
python scripts/evals/cswm_dmcontrol/render_pairs.py --env cartpole:swingup --horizon 40 --fps 20
python scripts/evals/cswm_dmcontrol/make_viewer.py
```

Outputs:
- `data/cswm_dmcontrol/pairs.jsonl`
- `results/runs/cswm_dmcontrol/<timestamp>/videos/*.mp4`
- `results/runs/cswm_dmcontrol/<timestamp>/summary.json`
- `results/runs/cswm_dmcontrol/<timestamp>/viewer/index.html`

### DreamerV3 / WM hook

`score_wm.py` is a stub. Once you have a DreamerV3 checkpoint for this DMControl env,
implement `DreamerV3Adapter.predict_latents(...)` (or imagined frames) and compute
WM separation vs GT divergence.

