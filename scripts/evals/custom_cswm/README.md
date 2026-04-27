## Custom CSWM (2 tasks) — H100 quick experiment

This creates a **2-task** evaluation using **your provided images**:

- **Task Door**: two images (door empty vs door blocked), same action "open the door"
  - QCM (A/B/C/D) over per-case outcomes.
- **Task Cup**: one image (cup near edge), two actions "move 5cm" vs "move 30cm"
  - QCM (A/B/C) for outcomes.

### 1) Put your images on the H100

Copy the files to the server, e.g.:

```bash
mkdir -p ~/CY/CSWM-Bench/data/custom_cswm/input
cp /path/to/door_empty.png ~/CY/CSWM-Bench/data/custom_cswm/input/door_empty.png
cp /path/to/door_box.png   ~/CY/CSWM-Bench/data/custom_cswm/input/door_box.png
cp /path/to/cup.png        ~/CY/CSWM-Bench/data/custom_cswm/input/cup.png
```

### 2) Build the dataset JSONL (copies images into dataset folder)

```bash
cd ~/CY/CSWM-Bench
python scripts/evals/custom_cswm/build_dataset.py \
  --door_empty data/custom_cswm/input/door_empty.png \
  --door_blocked data/custom_cswm/input/door_box.png \
  --cup data/custom_cswm/input/cup.png
```

Outputs:
- `data/custom_cswm/custom_cswm.jsonl`
- `data/custom_cswm/images/*.png`
- `reports/custom_cswm_viewer/index.html` (portable viewer)

### 3) Run a model (OpenAI/Gemini/Claude/OpenRouter vision)

```bash
export OPENAI_API_KEY="..."
python scripts/evals/custom_cswm/run_eval_custom_cswm.py --model gpt4o --max_samples 2
```

Results:
- `results/runs/custom_cswm/<timestamp>/<model>/results.json`
- `details.jsonl` (stores raw model outputs)

### Notes about DreamerV3 / JEPA-VL

DreamerV3 is not a text-VLM; it does **not** naturally accept "open the door" as an action.
To evaluate Dreamer/JEPA-VL, you need an adapter that maps:
  image(s) + action-text -> the model's actual action/state interface.

This package still helps because it pins down the **exact test format** (QCM + images)
and produces slide-ready artifacts while you wire up the adapter.

