"""
Object extraction from image using VLM + prompt.

The VLM sees the image and lists all visible objects. This list is then
used for open-vocabulary detection (OWL-ViT) to get bounding boxes.
No fixed object list — the VLM dynamically extracts what's in the image.
"""
import logging
import re
from typing import Callable, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

OBJECT_EXTRACTION_PROMPT = """List all distinct objects visible in this image. Include furniture, people, vehicles, animals, containers, and any other notable items.

Output ONLY a comma-separated list of object names. Use common English nouns (e.g. chair, table, person, car, dog, trash can, bottle).
Do not explain. Do not number. Just the list.
Example: chair, dining table, person, bottle, window"""


def extract_objects_from_image(
    image: Image.Image,
    generate_fn: Callable[[str, Image.Image, str], str],
    llm_name: str,
) -> List[str]:
    """
    Use VLM to extract object names from image. No fixed list — VLM sees
    the image and lists what's there.

    Args:
        image: PIL Image
        generate_fn: specialist_generate(llm_name, image, prompt) -> str
        llm_name: e.g. "qwen3_4b"

    Returns:
        List of object names, e.g. ["chair", "dining table", "person", "trash can"]
    """
    try:
        raw = generate_fn(llm_name, image, OBJECT_EXTRACTION_PROMPT)
        raw = (raw or "").strip()

        # Parse comma-separated list
        objects = []
        for part in re.split(r"[,;]\s*", raw):
            name = part.strip().lower()
            # Remove common prefixes/suffixes
            name = re.sub(r"^(the|a|an)\s+", "", name)
            name = re.sub(r"\s*\d+\.?\s*$", "", name)
            name = name.strip()
            if name and len(name) > 1 and name not in objects:
                objects.append(name)

        if not objects:
            logger.warning("Object extraction returned empty list. Raw: %s", raw[:200])
            return []

        return objects[:30]  # Cap to avoid OWL-ViT overload

    except Exception as e:
        logger.warning("Object extraction failed: %s", e)
        return []
