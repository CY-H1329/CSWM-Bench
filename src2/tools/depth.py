"""
Depth estimation tool for explicit_3d_representation agent.

Uses DepthAnythingV2 (HuggingFace) for monocular depth estimation.
Returns a textual summary of relative depth across image regions
suitable for injection into the 3D representation agent's prompt.
"""
import logging
from typing import Optional

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model
_depth_pipeline = None


def _get_depth_pipeline():
    """Lazy load DepthAnythingV2 pipeline."""
    global _depth_pipeline
    if _depth_pipeline is None:
        try:
            from transformers import pipeline
            # LiheYoung/depth-anything-small-hf is pipeline-compatible
            _depth_pipeline = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                trust_remote_code=True,
            )
            logger.info("DepthAnythingV2-Small loaded for depth tool")
        except Exception as e:
            logger.warning("Depth tool unavailable: %s. Returning placeholder.", e)
            _depth_pipeline = "unavailable"
    return _depth_pipeline


def get_depth_map(image: Image.Image) -> Optional[np.ndarray]:
    """
    Estimate depth and return the raw depth map (H×W array).

    Lower values = closer to viewer (typical for DepthAnything).
    Returns None on failure.
    """
    pipeline = _get_depth_pipeline()
    if pipeline == "unavailable":
        return None

    try:
        result = pipeline(image)
        depth_map = result.get("depth") or result.get("predicted_depth")
        if depth_map is None and isinstance(result, (np.ndarray, list)):
            depth_map = result
        if depth_map is None:
            return None

        if hasattr(depth_map, "numpy"):
            depth_arr = depth_map.numpy().squeeze()
        elif hasattr(depth_map, "cpu"):
            depth_arr = depth_map.cpu().numpy().squeeze()
        else:
            depth_arr = np.array(depth_map).squeeze()
        if depth_arr.ndim != 2:
            return None
        return depth_arr
    except Exception as e:
        logger.warning("Depth map extraction failed: %s", e)
        return None


def get_depth_summary(image: Image.Image) -> str:
    """
    Estimate depth from image and return a textual summary.

    Summary describes relative depth by region (closer/farther from viewer)
    so the explicit_3d_representation agent can use it for reasoning.

    Returns:
        Text block to inject into the agent prompt. On failure, returns
        a placeholder message.
    """
    depth_arr = get_depth_map(image)
    if depth_arr is None:
        return "[Depth tool unavailable. Proceed with visual analysis only.]"

    try:
        # Divide into 3x3 grid, compute mean depth per region
        h, w = depth_arr.shape
        region_h, region_w = h // 3, w // 3
        regions = []
        for i in range(3):
            for j in range(3):
                y0, y1 = i * region_h, (i + 1) * region_h if i < 2 else h
                x0, x1 = j * region_w, (j + 1) * region_w if j < 2 else w
                region = depth_arr[y0:y1, x0:x1]
                mean_val = float(np.nanmean(region))
                regions.append((i, j, mean_val))

        # Sort by depth (lower value = closer in most depth models)
        regions_sorted = sorted(regions, key=lambda r: r[2])
        pos_names = [
            ("top-left", 0, 0), ("top-center", 0, 1), ("top-right", 0, 2),
            ("mid-left", 1, 0), ("center", 1, 1), ("mid-right", 1, 2),
            ("bottom-left", 2, 0), ("bottom-center", 2, 1), ("bottom-right", 2, 2),
        ]
        pos_map = {(i, j): name for name, i, j in pos_names}

        closer = [pos_map[(r[0], r[1])] for r in regions_sorted[:3]]
        farther = [pos_map[(r[0], r[1])] for r in regions_sorted[-3:]]

        lines = [
            "## Depth Tool Output (relative depth from camera)",
            "",
            "Closer to viewer (lower depth): " + ", ".join(closer),
            "Farther from viewer (higher depth): " + ", ".join(farther),
            "",
            "Use this depth ordering to infer which objects are in front/behind.",
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.warning("Depth tool error: %s", e)
        return f"[Depth tool error: {e}. Proceed with visual analysis only.]"
