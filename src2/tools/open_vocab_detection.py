"""
Open-vocabulary object detection using OWL-ViT.

Takes image + list of object names (from VLM) and returns bounding boxes
for each detected instance. No fixed vocabulary — any object name works.
"""
import logging
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)

_owlvit_pipeline = None


def _get_owlvit_pipeline():
    """Lazy load OWL-ViT zero-shot detection pipeline."""
    global _owlvit_pipeline
    if _owlvit_pipeline is None:
        try:
            from transformers import pipeline
            _owlvit_pipeline = pipeline(
                task="zero-shot-object-detection",
                model="google/owlvit-base-patch32",
            )
            logger.info("OWL-ViT loaded for open-vocabulary detection")
        except Exception as e:
            logger.warning("OWL-ViT unavailable: %s", e)
            _owlvit_pipeline = "unavailable"
    return _owlvit_pipeline


def get_detections_with_labels(
    image: Image.Image,
    candidate_labels: List[str],
    threshold: float = 0.2,
) -> List[dict]:
    """
    Detect objects in image by text labels. Returns list of
    {id, label, box: (x1,y1,x2,y2), score}.

    Args:
        image: PIL Image
        candidate_labels: e.g. ["chair", "table", "person", "trash can"]
        threshold: minimum score (0-1)

    Returns:
        List of objects, sorted by score descending.
    """
    pipeline = _get_owlvit_pipeline()
    if pipeline == "unavailable" or not candidate_labels:
        return []

    try:
        # Pass as list to avoid "sequential on GPU" warning
        out = pipeline([image], candidate_labels=candidate_labels, threshold=threshold)
        if not out:
            predictions = []
        elif isinstance(out[0], list):
            predictions = out[0]  # list of images -> list of result lists
        else:
            predictions = out  # single image result as list of dicts

        objects = []
        for i, pred in enumerate(predictions or []):
            box_dict = pred.get("box", {})
            x1 = box_dict.get("xmin", 0)
            y1 = box_dict.get("ymin", 0)
            x2 = box_dict.get("xmax", 0)
            y2 = box_dict.get("ymax", 0)
            label = pred.get("label", "object")
            score = float(pred.get("score", 0))

            objects.append({
                "id": i + 1,
                "label": label,
                "box": (x1, y1, x2, y2),
                "score": score,
            })

        # Sort by score
        objects.sort(key=lambda o: o["score"], reverse=True)
        return objects[:25]  # Cap

    except Exception as e:
        logger.warning("OWL-ViT detection failed: %s", e)
        return []
