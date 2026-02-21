"""
SpatialReasoner Runner pour Spatial MAS.

Rôle MAS: Perception Agent — 3D / représentation spatiale explicite.
SpatialReasoner (NeurIPS'25): reasoning 3D spatial. Qwen2.5-VL backbone.
SOTA sur 3DSRBench.

Processor: base Qwen2.5-VL (ccvl a des problèmes de compat).

Refs:
- https://spatial-reasoner.github.io/
- https://huggingface.co/ccvl/SpatialReasoner
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS)

Requires: transformers>=4.50
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PIL import Image
import torch

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

from .base import BaseVLM

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/ccvl/SpatialReasoner
SPATIALREASONER_HF_URL = "https://huggingface.co/ccvl/SpatialReasoner"
SPATIALREASONER_MAS_ROLES = ["Direct", "3D", "SceneGraph"]


class SpatialReasonerRunner(BaseVLM):
    """
    Runner pour SpatialReasoner (ccvl/SpatialReasoner).

    HuggingFace: https://huggingface.co/ccvl/SpatialReasoner

    Perception Agent MAS: 3D reconstruction / représentation spatiale.
    Qwen2.5-VL based, reasoning 3D explicite.
    """

    def __init__(
        self,
        model_id: str = "ccvl/SpatialReasoner",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "SpatialReasoner requiert transformers>=4.50. "
                "pip install transformers>=4.50"
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else None
        dtype = torch_dtype or (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

        self.model_id = model_id
        self.device = device
        self.mas_roles = SPATIALREASONER_MAS_ROLES
        self.hf_url = SPATIALREASONER_HF_URL

        if "ccvl/SpatialReasoner" in model_id:
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )

        load_kwargs = dict(
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
        self.model.eval()

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
        Génère une réponse à partir de (image, prompt).
        Format Qwen2.5-VL: messages avec image + text.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        pad_id = (
            self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id
        )

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=pad_id,
            )

        in_len = inputs["input_ids"].shape[1]
        response = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        answer = response.strip()

        if return_full_output:
            return {
                "answer": answer,
                "raw": answer,
                "cot": self._extract_cot(answer),
            }
        return answer

    def _extract_cot(self, text: str) -> str:
        """Extrait le chain-of-thought pour shared memory (z_i)."""
        for marker in ["CoT:", "Reasoning:", "Strategy:", "3D representation:"]:
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
        """Rôle MAS préféré (3D spatial)."""
        return "3D"
