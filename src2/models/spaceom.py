"""
SpaceOm runner — remyxai/SpaceOm

Vision-language model for spatial reasoning.
Base: Qwen2.5-VL, fine-tuned with SpaceThinker/SpaceOm datasets.

Refs:
  - https://huggingface.co/remyxai/SpaceOm
  - 3DSRBench: 54.19%, BLINK: 59.9%, CV-Bench: 68.39%
"""
from typing import Optional
from PIL import Image
import torch

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

from .base import BaseVLM


class SpaceOmRunner(BaseVLM):
    """Runner for SpaceOm (remyxai/SpaceOm) — spatial reasoning VLM."""

    def __init__(
        self,
        model_id: str = "remyxai/SpaceOm",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "SpaceOm requires transformers with Qwen2.5-VL support. "
                "Install: pip install transformers>=4.45"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = device != "cpu" and torch.cuda.is_available()
        try:
            import accelerate  # noqa: F401
            has_accelerate = True
        except ImportError:
            has_accelerate = False

        if use_cuda and has_accelerate:
            load_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                **kwargs,
            )
        else:
            # Fallback when accelerate missing: load to CPU, then .to(device)
            load_kwargs = dict(
                torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                **{k: v for k, v in kwargs.items() if k not in ("device_map",)},
            )
        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if not (use_cuda and has_accelerate):
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device

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
        # Resize if large (SpaceOm example uses 512)
        if image_rgb.width > 1024 or image_rgb.height > 1024:
            ratio = image_rgb.height / image_rgb.width
            new_w = min(1024, image_rgb.width)
            new_h = int(new_w * ratio)
            image_rgb = image_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input],
            images=[image_rgb],
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )
        if top_p and top_p > 0:
            gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output.strip()
