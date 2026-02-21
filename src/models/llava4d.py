"""
LLaVA-4D Runner pour Spatial MAS.

Rôle MAS: Perception Agent — Direct / relation-focused.
LLaVA-4D (ICLR 2026) n'est pas encore public. Ce runner utilise LLaVA-1.6-NeXT
(llava-v1.6-mistral-7b-hf) comme proxy: même famille LLaVA, fort sur tâches spatiales/relation.

Refs:
- https://openreview.net/forum?id=URpbmVEsqB (LLaVA-4D)
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS)
- https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

Requires: transformers>=4.45 (LlavaNextForConditionalGeneration)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import AutoProcessor

try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextForConditionalGeneration = None

from .base import BaseVLM

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf (proxy LLaVA-4D)
LLAVA4D_HF_URL = "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA4D_MAS_ROLES = ["Direct", "3D", "SceneGraph"]


class LLaVA4DRunner(BaseVLM):
    """
    Runner pour LLaVA-4D (proxy: LLaVA-1.6-NeXT jusqu'à release).

    HuggingFace: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

    Perception Agent MAS: Direct / relation-focused.
    Format LLaVA: conversation avec image + text.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if LlavaNextForConditionalGeneration is None:
            raise ImportError(
                "LLaVA-4D (proxy) requiert transformers avec LlavaNext. "
                "pip install transformers>=4.45"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device
        dtype = torch_dtype or (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

        self.model_id = model_id
        self.device = device
        self.mas_roles = LLAVA4D_MAS_ROLES
        self.hf_url = LLAVA4D_HF_URL

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
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

        Format LLaVA: conversation avec content [image, text].
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_str = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        try:
            inputs = self.processor(image, prompt_str, return_tensors="pt").to(
                self.model.device
            )
        except TypeError:
            inputs = self.processor(
                images=[image], text=[prompt_str], padding=True, return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            **kwargs,
        )
        if temperature > 0:
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
            if top_p and top_p > 0:
                gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            start = inputs.input_ids.shape[1]
        else:
            start = 0

        answer = self.processor.decode(out[0][start:], skip_special_tokens=True)
        answer = answer.strip()

        if return_full_output:
            return {
                "answer": answer,
                "raw": answer,
                "cot": self._extract_cot(answer),
            }
        return answer

    def _extract_cot(self, text: str) -> str:
        """Extrait le chain-of-thought pour shared memory (z_i)."""
        for marker in ["CoT:", "Chain-of-thought:", "Reasoning:", "Strategy:"]:
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
        """Rôle MAS préféré (relation-focused)."""
        return "Direct"
