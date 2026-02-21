"""
Qwen3-VL-4B Runner for Spatial MAS.

Rôle MAS: Head-Agent (routing, task category, Top-3 selection) + Perception Agent (Direct 2D).
- Head: analyse query/image → predicted_category, selected_agents, coordination policy
- Perception: Direct Visual → analyse 2D directe de l'image

Refs:
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS, Beta credibility)
- https://arxiv.org/html/2502.18873v1 (MoSA, task routing)
- https://arxiv.org/html/2406.04692v1 (MoA, ensemble experts)

Requires: transformers>=4.51 (Qwen3VLForConditionalGeneration)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PIL import Image
import torch

try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        GenerationConfig,
    )
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    GenerationConfig = None

from .base import BaseVLM

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
QWEN3_HF_URL = "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
QWEN3_MAS_ROLES = ["Direct", "3D", "SceneGraph"]


class Qwen3Runner(BaseVLM):
    """
    Runner pour Qwen3-VL-4B-Instruct.

    HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

    Utilisé comme:
    - Head-Agent: routing task category, sélection Top-3, coordination policy
    - Perception Agent: rôle Direct (analyse 2D directe)

    Interface BaseVLM: generate(), run_batch()
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        use_flash_attn: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3-VL requiert transformers>=4.51. "
                "Install: pip install transformers>=4.51"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device

        dtype = torch_dtype or (
            torch.bfloat16 if device == "cuda" else torch.float32
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
                logger.debug("flash_attn non disponible, fallback attention par défaut")

        self.model_id = model_id
        self.device = device
        self.mas_roles = QWEN3_MAS_ROLES
        self.hf_url = QWEN3_HF_URL

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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

        Args:
            image: Image PIL
            prompt: Texte de la question / instruction
            temperature: 0 = déterministe
            max_new_tokens: Limite de tokens générés
            return_full_output: Si True, retourne {"answer": str, "raw": str, "cot": str}
                               pour extraction de z_i (shared memory)

        Returns:
            str ou dict selon return_full_output
        """
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_config.temperature = temperature
        gen_kwargs = {"generation_config": gen_config}
        for k, v in kwargs.items():
            if k not in ("top_k", "top_p", "temperature") and v is not None:
                gen_kwargs[k] = v

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs.get("input_ids")
        in_len = input_ids.shape[1] if input_ids is not None and hasattr(input_ids, "shape") else 0
        if hasattr(generated_ids, "shape"):
            generated_trimmed = [generated_ids[0][in_len:]]
        else:
            generated_trimmed = [generated_ids[in_len:]]

        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = output_text[0].strip() if output_text else ""

        if return_full_output:
            return {
                "answer": answer,
                "raw": answer,
                "cot": self._extract_cot(answer),
            }
        return answer

    def _extract_cot(self, text: str) -> str:
        """Extrait le chain-of-thought pour shared memory (z_i)."""
        for marker in ["CoT:", "Chain-of-thought:", "Reasoning:", "Justification:"]:
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
        """Exécute une série de (image, prompt) de manière séquentielle."""
        return [
            self.generate(img, p, **kwargs)
            for img, p in zip(images, prompts)
        ]

    def get_mas_role_preference(self) -> str:
        """Rôle MAS préféré pour ce modèle (Direct 2D)."""
        return "Direct"
