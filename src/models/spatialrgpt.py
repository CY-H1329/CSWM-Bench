"""
SpatialRGPT Runner pour Spatial MAS.

Rôle MAS: Perception Agent — 3D / SceneGraph / grounded spatial reasoning.
SpatialRGPT (NeurIPS'24): reasoning spatial ancré dans VLMs, backbone VILA 1.5.
Supporte region proposals + depth; pour VQA standard on utilise image-only.

Setup:
  git clone https://github.com/AnjieCheng/SpatialRGPT
  export SPATIALRGPT_PATH=/path/to/SpatialRGPT

Optionnel pour region/depth: DEPTH_ANYTHING_PATH, SAM_CKPT_PATH.

Refs:
- https://github.com/AnjieCheng/SpatialRGPT
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS)
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np
import torch

from .base import BaseVLM

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B
SPATIALRGPT_HF_URL = "https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B"
SPATIALRGPT_MAS_ROLES = ["Direct", "3D", "SceneGraph"]


def _get_spatialrgpt_path() -> Optional[str]:
    path = os.environ.get("SPATIALRGPT_PATH")
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    return None


def _load_via_spatialrgpt_repo(model_id: str, device: str, **kwargs):
    """Charge le modèle via le repo SpatialRGPT (load_pretrained_model)."""
    repo_path = _get_spatialrgpt_path()
    if not repo_path:
        raise ImportError(
            "SpatialRGPT requiert le repo officiel. "
            "Clonez-le et définissez SPATIALRGPT_PATH:\n"
            "  git clone https://github.com/AnjieCheng/SpatialRGPT\n"
            "  export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from llava.model.builder import load_pretrained_model

    model_name = "vila-siglip-llama-3b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id,
        model_name,
        device_map="auto" if device == "cuda" else None,
        device=device,
        **kwargs,
    )
    return tokenizer, model, image_processor, context_len


def _make_placeholder_depth(image: Image.Image) -> Image.Image:
    """Placeholder depth (gris) quand DepthAnything n'est pas disponible."""
    arr = np.array(image.convert("RGB"))
    gray = np.mean(arr, axis=-1).astype(np.uint8)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


class SpatialRGPTRunner(BaseVLM):
    """
    Runner pour SpatialRGPT (a8cheng/SpatialRGPT-VILA1.5-8B).

    HuggingFace: https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B

    Perception Agent MAS: 3D / SceneGraph.
    VILA 1.5 based, reasoning spatial avec depth/region optionnels.
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
        self.mas_roles = SPATIALRGPT_MAS_ROLES
        self.hf_url = SPATIALRGPT_HF_URL

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
        return_full_output: bool = False,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        Génère une réponse. VQA image-only (pas de regions).
        Format: <image>\\n{query}
        """
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        query = self._DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        images_tensor = self._process_images(
            [image_rgb], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        if self.model.config.get("enable_depth", False):
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
        answer = outputs.strip()

        if return_full_output:
            return {
                "answer": answer,
                "raw": answer,
                "cot": self._extract_cot(answer),
            }
        return answer

    def _extract_cot(self, text: str) -> str:
        """Extrait le chain-of-thought pour shared memory (z_i)."""
        for marker in ["CoT:", "Reasoning:", "Strategy:", "Log:"]:
            if marker in text:
                idx = text.find(marker) + len(marker)
                return text[idx:].strip()[:2000]
        return text[:1000]

    def run_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """Exécute une série de (image, prompt) séquentiellement."""
        return [
            self.generate(img, p, **kwargs)
            for img, p in zip(images, prompts)
        ]

    def get_mas_role_preference(self) -> str:
        """Rôle MAS préféré (SceneGraph / 3D)."""
        return "SceneGraph"
