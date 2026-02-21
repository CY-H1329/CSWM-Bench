"""
Scene Graph Tool — Grounding DINO + spatial relation graph.

1. Grounding DINO: open-vocabulary object detection
2. Build spatial relation graph from bbox positions (left_of, above, etc.)

Ref: https://huggingface.co/IDEA-Research/grounding-dino-tiny
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def _spatial_relation(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> str:
    """
    Infer spatial relation from bbox centers.
    Returns: left_of, right_of, above, below, overlapping, near
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2

    dx = cx2 - cx1
    dy = cy2 - cy1

    # Overlap check (simplified)
    def iou_1d(a1, a2, b1, b2):
        overlap = max(0, min(a2, b2) - max(a1, b1))
        return overlap / max(a2 - a1, b2 - b1, 1e-6)
    ix = iou_1d(bbox1[0], bbox1[2], bbox2[0], bbox2[2])
    iy = iou_1d(bbox1[1], bbox1[3], bbox2[1], bbox2[3])
    if ix > 0.3 and iy > 0.3:
        return "overlapping"

    # Horizontal
    if abs(dx) > abs(dy):
        return "left_of" if dx > 0 else "right_of"
    # Vertical
    return "above" if dy > 0 else "below"


class SceneGraphTool:
    """
    Build scene graph from image: objects + spatial relations.
    Uses Grounding DINO for detection.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        score_threshold: float = 0.3,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.score_threshold = score_threshold
        self._processor = None
        self._model = None

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
                if self.device == "cuda":
                    self._model = self._model.cuda()
                self._model.eval()
            except Exception as e:
                raise ImportError(
                    f"Scene graph tool failed to load {self.model_id}: {e}. "
                    "pip install transformers"
                ) from e

    def detect(
        self,
        image: Image.Image,
        text_prompt: Optional[List[List[str]]] = None,
    ) -> List[Dict]:
        """
        Detect objects. text_prompt: [["class1", "class2", ...]] for batch.
        Returns list of {label, score, box: [x1,y1,x2,y2]} (normalized 0-1).
        """
        self._ensure_loaded()
        import torch

        if text_prompt is None:
            text_prompt = [["object", "person", "car", "chair", "table", "bottle", "dog", "cat"]]

        image_rgb = image.convert("RGB") if image.mode != "RGB" else image
        inputs = self._processor(images=image_rgb, text=text_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([image_rgb.size[::-1]])
        if self.device == "cuda":
            target_sizes = target_sizes.cuda()
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.score_threshold,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )
        result = results[0] if results else {}

        out = []
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", result.get("text_labels", []))
        w, h = image_rgb.size
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if float(score) >= self.score_threshold:
                box_np = box.cpu().numpy() if hasattr(box, "cpu") else np.array(box)
                box_norm = [float(box_np[0]) / w, float(box_np[1]) / h, float(box_np[2]) / w, float(box_np[3]) / h]
                lbl = label if isinstance(label, str) else str(label)
                out.append({"label": lbl, "score": float(score), "box": box_norm})
        return out

    def build_graph(
        self,
        image: Image.Image,
        text_prompt: Optional[List[List[str]]] = None,
    ) -> str:
        """
        Build scene graph and return text representation for VLM.
        Format: "Objects: [obj1 at (x,y), ...]. Relations: obj1 left_of obj2, ..."
        """
        dets = self.detect(image, text_prompt)
        if not dets:
            return "Scene graph: No objects detected."

        # Build relation pairs
        nodes = []
        for i, d in enumerate(dets):
            box = d["box"]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            nodes.append((f"obj_{i}", d["label"], box, (cx, cy)))

        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                rel = _spatial_relation(nodes[i][2], nodes[j][2])
                edges.append((nodes[i][0], nodes[j][0], rel))

        obj_strs = [f"{label}({name}) at ({cx:.2f},{cy:.2f})" for name, label, _, (cx, cy) in nodes]
        rel_strs = [f"{a} {rel} {b}" for a, b, rel in edges[:20]]  # Limit for prompt size
        return (
            "Scene graph:\n"
            "Objects: " + "; ".join(obj_strs) + ".\n"
            "Relations: " + "; ".join(rel_strs) + "."
        )
