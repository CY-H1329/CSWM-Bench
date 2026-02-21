"""
Spatial MAS Tools — Depth, Scene Graph for role-specific perception.

- Depth: Depth Anything V2 (HuggingFace)
- Scene Graph: Grounding DINO + spatial relation graph
"""
from .depth import DepthTool
from .scene_graph import SceneGraphTool

__all__ = ["DepthTool", "SceneGraphTool"]
