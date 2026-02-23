"""
MAS v2 Tools — Role-specific augmentation for specialist agents.

C (Hybrid) approach:
  - direct_visual_heuristic: NO tools (pure pictorial cue reading)
  - explicit_3d_representation: Depth estimation tool
  - scene_graph_construction: Object detection + spatial relationship tool

Usage:
  from src2.tools import get_depth_summary, get_scene_graph_summary
  depth_text = get_depth_summary(image)      # for explicit_3d_representation
  graph_text = get_scene_graph_summary(image)  # for scene_graph_construction
"""
from .depth import get_depth_summary
from .scene_graph import get_scene_graph_summary

__all__ = ["get_depth_summary", "get_scene_graph_summary"]
