# Spatial MAS Tools

Role-specific tools for 3D and SceneGraph perception.

## Tools

### Depth (3D role)
- **Model**: [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- **Use**: `generate_as_3d(image, prompt)` → depth map + image concat → VLM
- **Requires**: `transformers>=4.45`

### Scene Graph (SceneGraph role)
- **Model**: [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- **Use**: `generate_as_scene_graph(image, prompt)` → object detection + spatial relations → text → VLM
- **Requires**: `transformers`

## Role Assignment (score-based)

Roles are **not** chosen by agents. They are assigned by `ScoreManager` based on s[agent, role, category]:

- `assign_roles_from_scores(score_manager, category, candidates)` → {role: agent}
- Pipeline passes `role` to `specialist_generate(agent, image, prompt, role=...)`
- Runner uses `generate_by_role(image, prompt, role)` → calls generate_as_3d / generate_as_scene_graph / generate_as_direct

## Usage

```python
from src.models import get_runner
from src.agents.mas import ScoreManager
from src.agents.mas.role_assignment import assign_roles_from_scores, agent_to_role_mapping

runner = get_runner("qwen3_4b", device="cuda")
score_mgr = ScoreManager()

# Role assignment
role_to_agent = assign_roles_from_scores(score_mgr, "depth", ["qwen3_4b", "sa2va", "spatialreasoner"])
agent_to_role = agent_to_role_mapping(role_to_agent)
# e.g. {"qwen3_4b": "Direct", "sa2va": "3D", "spatialreasoner": "SceneGraph"}

# Generate by role (with tools)
out = runner.generate_by_role(image, prompt, role="3D")  # uses Depth Anything V2
```
