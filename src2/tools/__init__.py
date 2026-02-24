"""
MAS v2 Tools — Role-specific augmentation for specialist agents.

C (Hybrid) approach:
  - direct_visual_heuristic: NO tools (pure pictorial cue reading)
  - explicit_3d_representation: 3D representation (VLM object extraction + OWL-ViT + depth)
  - scene_graph_construction: Object detection + spatial relationship tool

Usage:
  from src2.tools import get_3d_representation, extract_objects_from_image
  objects = extract_objects_from_image(image, specialist_generate, llm_name)
  text = get_3d_representation(image, object_names=objects)
"""
from .depth import get_depth_map, get_depth_summary
from .scene_graph import get_detected_objects, get_scene_graph_summary
from .representation_3d import get_3d_representation
from .object_extraction import extract_objects_from_image

__all__ = [
    "get_depth_map",
    "get_depth_summary",
    "get_detected_objects",
    "get_scene_graph_summary",
    "get_3d_representation",
    "extract_objects_from_image",
]
