"""
SpatialRGPT runner.

Model: a8cheng/SpatialRGPT-VILA1.5-8B
Backbone: VILA 1.5, supports optional depth/region proposals.

Requires the official SpatialRGPT repo:
  git clone https://github.com/AnjieCheng/SpatialRGPT
  export SPATIALRGPT_PATH=/path/to/SpatialRGPT

Refs:
  - https://github.com/AnjieCheng/SpatialRGPT
  - https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import List, Optional

from PIL import Image
import numpy as np
import torch

from .base import BaseVLM

logger = logging.getLogger(__name__)


def _get_spatialrgpt_path() -> Optional[str]:
    path = os.environ.get("SPATIALRGPT_PATH")
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    return None


def _load_via_spatialrgpt_repo(model_id: str, device: str, **kwargs):
    """Load model via the official SpatialRGPT repo (load_pretrained_model)."""
    repo_path = _get_spatialrgpt_path()
    if not repo_path:
        raise ImportError(
            "SpatialRGPT requires the official repo. "
            "Clone it and set SPATIALRGPT_PATH:\n"
            "  git clone https://github.com/AnjieCheng/SpatialRGPT\n"
            "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # Patch for transformers >= 4.40: no_init_weights was removed
    import transformers.modeling_utils as _mutils
    if not hasattr(_mutils, "no_init_weights"):
        from contextlib import nullcontext

        def _no_init_weights(_enable=True):
            if _enable:
                try:
                    from accelerate import init_empty_weights
                    return init_empty_weights()
                except ImportError:
                    return nullcontext()
            return nullcontext()

        _mutils.no_init_weights = _no_init_weights

    from llava.model.builder import load_pretrained_model

    # vila-siglip-llama3-8b for 8B model; vila-siglip-llama-3b for 3B
    model_name = "vila-siglip-llama3-8b" if "8b" in model_id.lower() or "8B" in model_id else "vila-siglip-llama-3b"
    # device_map=None: run without accelerate (load on CPU, then .to(device))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id,
        model_name,
        device_map=None,
        device=device,
        **kwargs,
    )
    return tokenizer, model, image_processor, context_len


def _make_placeholder_depth(image: Image.Image) -> Image.Image:
    """Placeholder grayscale depth when DepthAnything is not available."""
    arr = np.array(image.convert("RGB"))
    gray = np.mean(arr, axis=-1).astype(np.uint8)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


class SpatialRGPTRunner(BaseVLM):
    """Runner for SpatialRGPT (a8cheng/SpatialRGPT-VILA1.5-8B).

    VILA 1.5 based, grounded spatial reasoning with optional depth/region.
    For standard VQA: image-only mode (no region proposals).
    """

    def __init__(
        self,
        model_id: str = "a8cheng/SpatialRGPT-VILA1.5-8B",
        device: Optional[str] = None,
        conv_mode: str = "llama_3",
        use_depth: bool = False,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_id = model_id
        self.device = device
        self.conv_mode = conv_mode
        self.use_depth = use_depth

        tokenizer, model, image_processor, context_len = _load_via_spatialrgpt_repo(
            model_id, device, **kwargs
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        query = self._DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        images_tensor = self._process_images(
            [image_rgb], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        if getattr(self.model.config, "enable_depth", False):
            depth_img = _make_placeholder_depth(image_rgb)
            depths_tensor = self._process_images(
                [depth_img], self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)
        else:
            depths_tensor = None

        input_ids = (
            self._tokenizer_image_token(
                full_prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        stopping_criteria = self._KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        gen_kwargs = dict(
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        if top_p and top_p > 0:
            gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k

        images_list = [images_tensor]
        depths_list = [depths_tensor] if depths_tensor is not None else [images_tensor]
        masks_list = [None]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*")
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_list,
                    depths=depths_list,
                    masks=masks_list,
                    **gen_kwargs,
                )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()
