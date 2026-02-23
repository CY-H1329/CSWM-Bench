"""
SpatialReasoner runner.

Model: ccvl/SpatialReasoner
Backbone: Qwen2.5-VL (uses Qwen/Qwen2.5-VL-7B-Instruct processor due to compat issues)
SOTA on 3DSRBench.

Refs:
  - https://spatial-reasoner.github.io/
  - https://huggingface.co/ccvl/SpatialReasoner

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


class SpatialReasonerRunner(BaseVLM):
    """Runner for SpatialReasoner (ccvl/SpatialReasoner).

    Qwen2.5-VL based, specialised in 3D spatial reasoning.
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
                "SpatialReasoner requires transformers>=4.50. "
                "pip install transformers>=4.50"
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch_dtype or (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

        self.model_id = model_id
        self.device = device

        # ccvl/SpatialReasoner has processor compat issues; use base Qwen2.5-VL processor
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
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            **kwargs,
        )
        # No device_map — run without accelerate
        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.model = self.model.to(device)
        self.model.eval()

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
        return response.strip()
