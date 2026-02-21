"""
Sa2VA-4B Runner pour Spatial MAS.

Rôle MAS: Perception Agent — spécialiste spatial (depth, distance).
ByteDance Sa2VA: modèle vision-language avec prédiction forward pour chat image.
Format: <image> + text, predict_forward() pour inférence.

Refs:
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS, Beta credibility)
- https://arxiv.org/html/2406.04692v1 (MoA, ensemble experts)

Requires: transformers, trust_remote_code=True
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from .base import BaseVLM

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/ByteDance/Sa2VA-4B
SA2VA_HF_URL = "https://huggingface.co/ByteDance/Sa2VA-4B"
SA2VA_MAS_ROLES = ["Direct", "3D", "SceneGraph"]


def _patch_tied_weights_for_sa2va():
    """Sa2VA utilise _tied_weights_keys; transformers récent attend all_tied_weights_keys."""
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self):
        if not hasattr(self, "all_tied_weights_keys"):
            old = getattr(self, "_tied_weights_keys", None)
            if old is not None and hasattr(old, "keys"):
                self.all_tied_weights_keys = old
            elif isinstance(old, (list, tuple)):
                self.all_tied_weights_keys = {
                    k: None
                    for x in old
                    for k in (x if isinstance(x, (list, tuple)) else [x])
                }
            else:
                self.all_tied_weights_keys = {}
        _orig(self)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched


def _patch_torch_linspace_for_sa2va():
    """InternVisionModel utilise torch.linspace().item() qui échoue sur meta tensors.
    Force CPU device pour éviter meta device de transformers/accelerate."""
    _orig = torch.linspace

    def _patched(*args, **kwargs):
        kwargs.setdefault("device", torch.device("cpu"))
        return _orig(*args, **kwargs)

    torch.linspace = _patched
    return _orig


class Sa2VARunner(BaseVLM):
    """
    Runner pour Sa2VA (ByteDance/Sa2VA-4B).

    HuggingFace: https://huggingface.co/ByteDance/Sa2VA-4B

    Perception Agent MAS: spécialiste depth/distance.
    Utilise model.predict_forward() pour le chat image.
    """

    def __init__(
        self,
        model_id: str = "ByteDance/Sa2VA-4B",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch_dtype or torch.bfloat16

        self.model_id = model_id
        self.device = device
        self.mas_roles = SA2VA_MAS_ROLES
        self.hf_url = SA2VA_HF_URL

        load_kwargs = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            device_map=None,
            **kwargs,
        )

        _patch_tied_weights_for_sa2va()
        _orig_linspace = _patch_torch_linspace_for_sa2va()
        try:
            self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        finally:
            torch.linspace = _orig_linspace

        if device == "cuda":
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

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
        Génère une réponse via predict_forward.

        Format Sa2VA: <image> + text.
        predict_forward retourne un dict avec 'prediction'.
        """
        text_prompts = f"<image>{prompt}"
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        input_dict = {
            "image": image_rgb,
            "text": text_prompts,
            "past_text": "",
            "mask_prompts": None,
            "tokenizer": self.tokenizer,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Passing `generation_config` together with generation-related arguments",
            )
            return_dict = self.model.predict_forward(**input_dict)

        answer = (return_dict.get("prediction") or "").strip()

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
        """Rôle MAS préféré (depth/distance specialist)."""
        return "3D"
