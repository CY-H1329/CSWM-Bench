# CV-Bench & 3DSRBench Results Summary

Generated: 2026-02-18 09:15

---

## CV-Bench

### Overall Accuracy

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 61.90% | 28.43% |
| llava4d_20260217_154038 | - | 56.67% |
| llava4d_20260217_154714 | - | 53.33% |
| llava4d_full_dataset | 61.90% | 28.43% |
| qwen3_4b | 81.35% | 83.78% |
| qwen3_4b_20260217_153223 | - | 86.67% |
| qwen3_4b_20260217_154328 | - | 90.00% |
| qwen3_4b_full_dataset | 81.35% | 83.78% |
| sa2va | 69.14% | 42.80% |
| sa2va_20260217_153743 | - | 66.67% |
| sa2va_20260217_154618 | - | 50.00% |
| sa2va_full_dataset | 69.14% | 42.80% |

### CV-Bench by Category (Count, Relation, Depth, Distance)

#### Count

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 53.81% | 60.15% |
| qwen3_4b | 64.72% | 65.23% |
| sa2va | 57.87% | 40.10% |

#### Relation

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 67.23% | 14.46% |
| qwen3_4b | 87.85% | 94.00% |
| sa2va | 78.00% | 7.69% |

#### Depth

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 71.33% | 5.67% |
| qwen3_4b | 93.83% | 95.67% |
| sa2va | 79.83% | 80.00% |

#### Distance

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 57.33% | 24.67% |
| qwen3_4b | 83.67% | 85.17% |
| sa2va | 63.67% | 47.17% |

---

## 3DSRBench

### Overall Accuracy

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| claude_sonnet_4_5 | 66.60% | - |
| gemini_robotics_er | 64.20% | 42.90% |
| gpt4o | 39.00% | 58.80% |
| llava4d_20260215_072543 | - | 52.00% |
| llava4d_20260215_080404 | - | 50.00% |
| llava4d_20260216_081426 | - | 49.64% |
| llava4d_full_dataset | - | 31.12% |
| qwen3_4b_20260215_070758 | - | 68.00% |
| qwen3_4b_20260215_081632 | - | 68.00% |
| qwen3_4b_20260215_082913 | - | 60.62% |
| qwen3_4b_full_dataset | - | 60.58% |
| sa2va_20260215_071929 | - | 56.00% |
| sa2va_20260215_081033 | - | 60.00% |
| sa2va_20260215_235003 | - | 52.12% |
| sa2va_full_dataset | - | 22.44% |

### 3DSRBench by Category (12 categories)

| Category | claude_sonnet_4 | claude_sonnet_4 | gemini_robotics | gemini_robotics | gpt4o |
|---|---|---|---|---|---|
| location_above | 73.08% | - | 70.00% | 68.46% | 69.23% |
| height_higher | 67.41% | - | 59.26% | 11.85% | 51.85% |
| location_closer_to_camera | 78.42% | - | 79.86% | 20.14% | 66.91% |
| multi_object_closer_to | 72.00% | - | 69.33% | 44.00% | 68.00% |
| orientation_on_the_left | 56.36% | - | 54.55% | 58.18% | 63.64% |
| multi_object_facing | 62.50% | - | 57.81% | 40.62% | 42.19% |

*Full category table in category_performance.csv*
