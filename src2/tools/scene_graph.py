"""
Scene graph tool for scene_graph_construction agent.

Uses object detection (DETR or similar) to extract objects and bounding boxes,
then computes pairwise spatial relationships (above/below, left/right, overlap).
Returns a textual summary suitable for injection into the scene graph agent's prompt.
"""
import logging
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model
_detector = None
_COCO_LABELS = None  # DETR uses COCO 91 classes


def _get_detector():
    """Lazy load object detector."""
    global _detector, _COCO_LABELS
    if _detector is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            _detector = (proc, model)
            # COCO class names (DETR 91 classes, index 0 = background)
            _COCO_LABELS = [
                "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
                "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
                "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
                "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush",
            ]
            logger.info("DETR-ResNet50 loaded for scene graph tool")
        except Exception as e:
            logger.warning("Scene graph tool unavailable: %s. Returning placeholder.", e)
            _detector = "unavailable"
    return _detector


def _compute_iou(box_a: Tuple[float, float, float, float],
                 box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _get_spatial_relation(box_a: Tuple[float, float, float, float],
                         box_b: Tuple[float, float, float, float],
                         iou: float) -> List[str]:
    """Infer spatial relations from bounding boxes."""
    rels = []
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2

    if iou > 0.05:
        rels.append("overlaps (possible occlusion)")
    if cy_a < cy_b - 5:
        rels.append("above")
    elif cy_a > cy_b + 5:
        rels.append("below")
    if cx_a < cx_b - 5:
        rels.append("left_of")
    elif cx_a > cx_b + 5:
        rels.append("right_of")

    return rels if rels else ["adjacent"]


def get_detected_objects(image: Image.Image, n_keep: int = 15) -> List[dict]:
    """
    Detect objects in image and return list of {id, label, box, score}.

    box is (x1, y1, x2, y2) in image coordinates.
    Returns empty list on failure.
    """
    detector = _get_detector()
    if detector == "unavailable":
        return []

    processor, model = detector
    global _COCO_LABELS
    labels = _COCO_LABELS or []

    try:
        import torch
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5,
        )
        if not results:
            return []
        result = results[0]

        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        pred_labels = result["labels"].cpu().numpy()

        n_keep = min(n_keep, len(boxes))
        if n_keep == 0:
            return []

        objects = []
        id2label = getattr(model.config, "id2label", None) or {}
        for i in range(n_keep):
            x1, y1, x2, y2 = boxes[i].tolist()
            lid = int(pred_labels[i])
            label = id2label.get(lid, labels[lid] if lid < len(labels) else "object")
            if not label or label == "N/A":
                label = "object"
            objects.append({
                "id": i + 1,
                "label": label,
                "box": (x1, y1, x2, y2),
                "score": float(scores[i]),
            })
        return objects

    except Exception as e:
        logger.warning("Scene graph detection failed: %s", e)
        return []


def get_scene_graph_summary(image: Image.Image) -> str:
    """
    Detect objects and compute spatial relationships.

    Returns a textual summary: objects with positions and pairwise
    relationships (above/below, left/right, overlaps).
    """
    objects = get_detected_objects(image)
    if not objects:
        return "[Scene graph tool unavailable or no objects detected. Proceed with visual analysis only.]"

    try:
        # Build relationship summary
        lines = [
            "## Scene Graph Tool Output",
            "",
            "### Detected Objects (id, label, position in image)",
        ]
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pos = "upper" if cy < image.height / 3 else ("lower" if cy > 2 * image.height / 3 else "middle")
            pos += "-left" if cx < image.width / 3 else ("-right" if cx > 2 * image.width / 3 else "-center")
            lines.append(f"  {obj['id']}. {obj['label']} ({pos}, score={obj['score']:.2f})")

        lines.append("")
        lines.append("### Pairwise Spatial Relationships")

        for i, oa in enumerate(objects):
            for j, ob in enumerate(objects):
                if i >= j:
                    continue
                iou = _compute_iou(oa["box"], ob["box"])
                rels = _get_spatial_relation(oa["box"], ob["box"], iou)
                if rels:
                    rel_str = ", ".join(rels)
                    lines.append(f"  {oa['label']}({oa['id']}) — {rel_str} — {ob['label']}({ob['id']})")

        lines.append("")
        lines.append("Use this graph to reason about the question.")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("Scene graph tool error: %s", e)
        return f"[Scene graph tool error: {e}. Proceed with visual analysis only.]"
