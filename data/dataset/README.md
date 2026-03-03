# Train Datasets

Training datasets disjoint from frozen benchmarks (eval).

| Dataset | Samples | Source | Excludes |
|---------|---------|--------|----------|
| **cvbench_train_300** | 300 | CV-Bench test | cvbench_400 (frozen) |
| **3dsrbench_train_300** | 300 | 3DSRBench benchmark | 3dsrbench_500 (frozen) |
| **stvqa_train_300** | 300 | STVQA-7K train | - (frozen uses val) |

## Usage

```python
from datasets import load_from_disk

ds = load_from_disk("data/dataset/cvbench_train_300")
```

## Regenerate

```bash
python scripts/prepare_train_datasets.py
```
