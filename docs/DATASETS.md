# Datasets

This project uses the following benchmarks for spatial reasoning evaluation.

## 3DSRBench

- **HuggingFace**: [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench)
- **Subset**: `benchmark`
- **Split**: `test`
- **Format**: Multiple choice (A/B/C/D), 12 fine-grained categories
- **Samples**: ~5.1k

### Categories (12)

- location_above, height_higher, location_closer_to_camera
- multi_object_closer_to, orientation_on_the_left, multi_object_facing
- multi_object_same_direction, orientation_in_front_of
- multi_object_viewpoint_towards_object, orientation_viewpoint
- location_next_to, multi_object_parallel

### Usage

```bash
# GPU models (Qwen3, Sa2VA, LLaVA4D)
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py --full_dataset
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py --full_dataset

# API models (Claude, GPT-4o, Gemini)
python scripts/evals/3dsrbench_api/run_eval_api.py --full_dataset
```

---

## CV-Bench

- **HuggingFace**: [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
- **Split**: `test`
- **Format**: Multiple choice (choices)
- **Samples**: ~2.6k

### Usage

```bash
python run_eval_mas.py --benchmark cvbench --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b
```

---

## OMNI3D-BENCH

- **HuggingFace**: [dmarsili/Omni3D-Bench](https://huggingface.co/datasets/dmarsili/Omni3D-Bench)
- **Split**: `train`
- **Format**: Free-form answer
- **Samples**: ~501

### Usage

```bash
python run_eval_mas.py --benchmark omni3d --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b
```

---

## GQA (planned)

GQA may be added in future experiments. Check the repository for updates.
