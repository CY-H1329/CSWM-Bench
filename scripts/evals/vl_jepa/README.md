## VL-JEPA quick test on CSWM image pairs

You asked: “use these images to test VL-JEPA + DreamerV3”.

### What we can do reliably with VL-JEPA (today)

VL-JEPA is a **visual representation** model. A clean, direct test on your
CSWM pairs is:

- compute image embeddings for `case1` and `case2`
- measure cosine distance
- check whether the model **separates** near-identical scenes when the
  physical consequence differs (this supports your motivation: representation
  blind spots)

This does NOT require video generation.

### What we cannot do reliably with DreamerV3 on these photos

DreamerV3 is an **action-conditioned world model** trained inside an
environment. It does not accept arbitrary real photos + text actions.

To evaluate DreamerV3, you need:
- an environment with observations (images) and **numeric actions**
- a trained DreamerV3 checkpoint for that environment
- then run counterfactual rollouts for action pairs and compare divergence

See `scripts/evals/world_models/dreamerv3_notes.md`.

### Run the VL-JEPA embedding separation script

First make sure `data/cswmbench/cswmbench.jsonl` exists:

```bash
python scripts/evals/cswmbench_api/generate_cswmbench.py
```

Then run:

```bash
python scripts/evals/vl_jepa/run_embed_separation.py \
  --data data/cswmbench/cswmbench.jsonl \
  --out results/runs/vl_jepa_embed
```

If you have an actual VL-JEPA checkpoint / repo, use:

```bash
python scripts/evals/vl_jepa/run_embed_separation.py \
  --data data/cswmbench/cswmbench.jsonl \
  --encoder custom \
  --custom_loader /path/to/your_vljepa_loader.py \
  --out results/runs/vl_jepa_embed
```

