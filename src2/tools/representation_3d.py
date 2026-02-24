"""
3D Representation tool for explicit_3d_representation agent.

Combines DepthAnything (depth map) + object detection to produce
object-level depth ordering. Objects can come from:
  - VLM-extracted list + OWL-ViT (open-vocabulary, no fixed list)
  - DETR (fallback, COCO 80 classes)

Each detected object gets a depth value from the mean depth within
its bounding box, enabling direct answers to "which is closer?",
"is A in front of B?", "how many X?" etc.
"""
import logging
from collections import Counter
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np

from .depth import get_depth_map
from .open_vocab_detection import get_detections_with_labels
from .scene_graph import get_detected_objects

logger = logging.getLogger(__name__)

_POS_NAMES = [
    ("top-left", 0, 0), ("top-center", 0, 1), ("top-right", 0, 2),
    ("mid-left", 1, 0), ("center", 1, 1), ("mid-right", 1, 2),
    ("bottom-left", 2, 0), ("bottom-center", 2, 1), ("bottom-right", 2, 2),
]


def _bbox_to_position(box: Tuple[float, float, float, float],
                      img_w: int, img_h: int) -> str:
    """Convert bbox center to region name (e.g. bottom-center)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    row = 0 if cy < img_h / 3 else (2 if cy > 2 * img_h / 3 else 1)
    col = 0 if cx < img_w / 3 else (2 if cx > 2 * img_w / 3 else 1)
    pos_map = {(r, c): name for name, r, c in _POS_NAMES}
    return pos_map.get((row, col), "center")


def _mean_depth_in_bbox(depth_arr: np.ndarray, box: Tuple[float, float, float, float],
                       img_w: int, img_h: int) -> float:
    """Compute mean depth within bbox. Scales bbox to depth map size."""
    h_d, w_d = depth_arr.shape
    x1, y1, x2, y2 = box
    # Scale from image coords to depth map coords
    x1_s = int(x1 * w_d / img_w)
    y1_s = int(y1 * h_d / img_h)
    x2_s = int(x2 * w_d / img_w)
    y2_s = int(y2 * h_d / img_h)
    x1_s = max(0, min(x1_s, w_d - 1))
    y1_s = max(0, min(y1_s, h_d - 1))
    x2_s = max(x1_s + 1, min(x2_s, w_d))
    y2_s = max(y1_s + 1, min(y2_s, h_d))
    region = depth_arr[y1_s:y2_s, x1_s:x2_s]
    return float(np.nanmean(region))


def get_3d_representation(
    image: Image.Image,
    object_names: Optional[List[str]] = None,
) -> str:
    """
    Build object-level 3D representation: depth map + object detection
    → each object gets mean depth in its bbox → sorted depth ordering.

    Args:
        image: PIL Image
        object_names: If provided, use OWL-ViT with these labels (from VLM).
                      If None, fallback to DETR (COCO 80 classes).

    Returns a textual summary for the explicit_3d_representation agent.
    Lower depth value = closer to viewer (DepthAnything convention).
    """
    depth_arr = get_depth_map(image)
    if object_names:
        objects = get_detections_with_labels(image, candidate_labels=object_names)
    else:
        objects = get_detected_objects(image, n_keep=15)

    if depth_arr is None:
        return "[3D tool: Depth estimation failed. Proceed with visual analysis only.]"
    if not objects:
        return "[3D tool: No objects detected. Proceed with visual analysis only.]"

    try:
        img_w, img_h = image.size
        h_d, w_d = depth_arr.shape

        # Compute mean depth per object
        for obj in objects:
            obj["mean_depth"] = _mean_depth_in_bbox(
                depth_arr, obj["box"], img_w, img_h
            )
            obj["position"] = _bbox_to_position(obj["box"], img_w, img_h)

        # Sort by depth: lower = closer
        objects_sorted = sorted(objects, key=lambda o: o["mean_depth"])
        d_min = objects_sorted[0]["mean_depth"]
        d_max = objects_sorted[-1]["mean_depth"]
        d_range = max(d_max - d_min, 1e-6)

        # Normalized depth [0, 1]: 0 = closest, 1 = farthest
        for obj in objects_sorted:
            obj["depth_norm"] = (obj["mean_depth"] - d_min) / d_range

        # --- Depth Map Grid (3×3, mathematical) ---
        region_h, region_w = h_d // 3, w_d // 3
        depth_grid = []
        for i in range(3):
            row_vals = []
            for j in range(3):
                y0 = i * region_h
                y1 = (i + 1) * region_h if i < 2 else h_d
                x0 = j * region_w
                x1 = (j + 1) * region_w if j < 2 else w_d
                region = depth_arr[y0:y1, x0:x1]
                val = float(np.nanmean(region))
                row_vals.append(val)
            depth_grid.append(row_vals)
        grid_min = min(v for row in depth_grid for v in row)
        grid_max = max(v for row in depth_grid for v in row)
        grid_range = max(grid_max - grid_min, 1e-6)

        # Build output
        lines = [
            "## 3D Representation Tool Output (mathematical depth)",
            "",
            "### 1. Depth Map Grid (3×3, normalized [0=closest, 1=farthest])",
            "  Image regions with relative depth. Lower value → closer to camera.",
            "",
        ]

        pos_names = [
            ["top-left", "top-center", "top-right"],
            ["mid-left", "center", "mid-right"],
            ["bottom-left", "bottom-center", "bottom-right"],
        ]
        for i, row in enumerate(depth_grid):
            norm_vals = [(v - grid_min) / grid_range for v in row]
            row_str = "  ".join(f"{pos_names[i][j]}:{norm_vals[j]:.2f}" for j in range(3))
            lines.append(f"  {row_str}")

        lines.append("")
        lines.append("### 2. Object Depth Values (normalized z ∈ [0, 1], 0=closest)")
        for rank, obj in enumerate(objects_sorted, 1):
            pos = obj["position"]
            label = obj["label"]
            z = obj["depth_norm"]
            suffix = " [CLOSEST]" if rank == 1 else (" [FARTHEST]" if rank == len(objects_sorted) else "")
            lines.append(f"  {rank}. {label} ({pos}): z={z:.3f}{suffix}")

        lines.append("")
        lines.append("### 3. Depth Ordering (adjacent pairs, z_A < z_B ⇒ A in front)")

        for i in range(len(objects_sorted) - 1):
            oa, ob = objects_sorted[i], objects_sorted[i + 1]
            z_a, z_b = oa["depth_norm"], ob["depth_norm"]
            delta_z = z_b - z_a
            lines.append(f"  - {oa['label']} (z={z_a:.2f}) → {ob['label']} (z={z_b:.2f})  [Δz={delta_z:.2f}]")

        # Count per label (for "how many X?" questions)
        label_counts = Counter(o["label"] for o in objects_sorted)
        lines.append("")
        lines.append("### 4. Instance Count by Object Type")
        for label, cnt in label_counts.most_common():
            lines.append(f"  - {label}: {cnt}")

        lines.append("")
        lines.append("### Mathematical Interpretation")
        lines.append("  - z: normalized depth (0=closest to camera, 1=farthest). Monocular depth, relative scale.")
        lines.append("  - Δz: depth difference. Larger Δz = greater separation in depth.")
        lines.append("  - Depth Map Grid: spatial layout. Compare object position to grid values.")

        lines.append("")
        lines.append("### How to Use")
        lines.append("  - 'Which closer?': pick lower z (or lower rank).")
        lines.append("  - 'A in front of B?': check Pairwise; z_A < z_B means A in front.")
        lines.append("  - 'How many X?': use Instance Count.")
        lines.append("  - Trust z values over pictorial cues when available.")

        return "\n".join(lines)

    except Exception as e:
        logger.warning("3D representation tool error: %s", e)
        return f"[3D tool error: {e}. Proceed with visual analysis only.]"
