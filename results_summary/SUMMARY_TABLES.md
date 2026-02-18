# CV-Bench & 3DSRBench Results Summary

Generated: 2026-02-18 18:19

---

## CV-Bench

### Overall Accuracy

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| llava4d | 61.90% | 28.43% |
| qwen3_4b | 81.35% | 83.78% |
| sa2va | 69.14% | 42.80% |

## 3DSRBench

### Overall Accuracy

| Model | With Prompt | Without Prompt |
|-------|-------------|----------------|
| claude_sonnet_4_5 | 66.60% | - |
| gemini_robotics_er | 64.20% | 42.90% |
| gpt4o | 39.00% | 58.80% |
| llava4d | - | 31.12% |
| qwen3_4b | - | 60.58% |
| sa2va | - | 22.44% |

### 3DSRBench by Category (12 categories)

| Category | claude(wp) | claude(wop) | gemini(wp) | gemini(wop) | gpt4o(wp) | gpt4o(wop) |
|---|---|---|---|---|---|---|
| location_above | 73.08% | - | 70.00% | 68.46% | 52.31% | 69.23% |
| height_higher | 67.41% | - | 59.26% | 11.85% | 24.44% | 51.85% |
| location_closer_to_camera | 78.42% | - | 79.86% | 20.14% | 44.60% | 66.91% |
| multi_object_closer_to | 72.00% | - | 69.33% | 44.00% | 46.67% | 68.00% |
| orientation_on_the_left | 56.36% | - | 54.55% | 58.18% | 40.00% | 63.64% |
| multi_object_facing | 62.50% | - | 57.81% | 40.62% | 32.81% | 42.19% |
| multi_object_same_direction | 60.87% | - | 53.62% | 57.97% | 34.78% | 47.83% |
| orientation_in_front_of | 77.14% | - | 68.57% | 57.14% | 50.00% | 64.29% |
| multi_object_viewpoint_towar | 35.38% | - | 30.77% | 26.15% | 21.54% | 38.46% |
| orientation_viewpoint | 45.16% | - | 50.00% | 45.16% | 25.81% | 50.00% |
| location_next_to | 78.87% | - | 80.28% | 59.15% | 40.85% | 74.65% |
| multi_object_parallel | 66.15% | - | 73.85% | 58.46% | 47.69% | 53.85% |

*Full table: category_performance.csv*
