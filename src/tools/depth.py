"""
Depth Tool — Depth Anything V2 for 3D role.

Uses HuggingFace transformers pipeline.
Model: depth-anything/Depth-Anything-V2-Small-hf (24.8M, fast)
Alternative: Depth-Anything-V2-Base-hf, Depth-Anything-V2-Large-hf

Ref: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

_DEPTH_PIPELINE = None


def _get_depth_pipeline(device: int = -1):
    """Lazy load depth pipeline."""
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is None:
        try:
            from transformers import pipeline
            _DEPTH_PIPELINE = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device,
            )
        except Exception as e:
            logger.warning("Depth Anything V2 load failed: %s", e)
            raise ImportError(
                "Depth tool requires: pip install transformers>=4.45. "
                "Depth-Anything-V2-Small-hf will be downloaded."
            ) from e
    return _DEPTH_PIPELINE


class DepthTool:
    """
    Depth estimation via Depth Anything V2.
    Returns depth map as PIL Image (grayscale, normalized).
    """

    def __init__(self, device: int = -1, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.device = device
        self.model_id = model_id
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
                self._pipe = pipeline(
                    task="depth-estimation",
                    model=self.model_id,
                    device=self.device,
                )
            except Exception as e:
                raise ImportError(
                    f"Depth tool failed to load {self.model_id}: {e}. "
                    "pip install transformers>=4.45"
                ) from e

    def estimate(self, image: Image.Image) -> Image.Image:
        """
        Estimate depth map from image.
        Returns PIL Image (grayscale, 0=far, 255=near).
        """
        self._ensure_loaded()
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image
        out = self._pipe(image_rgb)
        depth = out.get("depth") or out.get("depth_map")
        if depth is None:
            # Some pipelines return different keys
            depth = list(out.values())[0] if out else None
        if depth is None:
            raise RuntimeError("Depth pipeline returned no depth map")
        if isinstance(depth, np.ndarray):
            # Normalize to 0-255 for PIL
            dmin, dmax = depth.min(), depth.max()
            if dmax > dmin:
                depth = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                depth = np.zeros_like(depth, dtype=np.uint8)
            return Image.fromarray(depth, mode="L")
        return depth

    def estimate_and_concat(self, image: Image.Image) -> Image.Image:
        """
        Return RGB image with depth as 4th channel or side-by-side.
        For models that accept multi-channel: [R,G,B,Depth].
        Fallback: horizontal concat [image | depth_colored].
        """
        depth = self.estimate(image)
        # Colorize depth for visualization (viridis-like)
        depth_np = np.array(depth)
        depth_color = _colorize_depth(depth_np)
        depth_pil = Image.fromarray(depth_color)
        # Horizontal concat: original | depth
        w, h = image.size
        out = Image.new("RGB", (w * 2, h))
        out.paste(image.convert("RGB"), (0, 0))
        out.paste(depth_pil, (w, 0))
        return out


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Simple colormap: blue (far) -> red (near)."""
    dmin, dmax = depth.min(), depth.max()
    if dmax <= dmin:
        return np.stack([depth, depth, depth], axis=-1)
    t = (depth.astype(float) - dmin) / (dmax - dmin)
    # Blue -> Cyan -> Green -> Yellow -> Red
    r = np.clip(4 * t - 2, 0, 1) + np.clip(2 - 4 * t, 0, 1) * (t < 0.5)
    g = np.clip(4 * t - 1, 0, 1) * (t >= 0.25) * (t < 0.75) + (t >= 0.75)
    b = 1 - np.clip(4 * t, 0, 1)
    rgb = np.stack([(r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)], axis=-1)
    return rgb
