# MAS v2 Tools (C Hybrid Approach)

Role-specific tools augment specialist agents. Only 2 of 3 roles use tools:

| Role | Tool | Model |
|------|------|-------|
| `direct_visual_heuristic` | **None** | Pure pictorial cue reading |
| `explicit_3d_representation` |  Depth | LiheYoung/depth-anything-small-hf |
| `scene_graph_construction` | Scene graph | facebook/detr-resnet-50 |

## Usage

```python
from src2.tools import get_depth_summary, get_scene_graph_summary
from PIL import Image

image = Image.open("scene.jpg")
depth_text = get_depth_summary(image)      # For explicit_3d agent
graph_text = get_scene_graph_summary(image)  # For scene_graph agent
```

## Dependencies

- `transformers` (already in requirements.txt)
- `torch` (already in requirements.txt)

No extra pip install needed. Tools lazy-load on first use.

## Output Format

- **Depth**: Text block describing relative depth by region (closer/farther)
- **Scene graph**: Detected objects + pairwise spatial relationships (above/below, left/right, overlaps)

## Integration

The pipeline (`run_step`) automatically runs tools for `explicit_3d_representation` and `scene_graph_construction` and injects the output into their prompts. No manual tool calls needed.
