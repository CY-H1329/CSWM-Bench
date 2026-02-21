"""
LLaVA-4D inference for MAS Perception Agent.

LLaVA-4D (ICLR 2026) is not yet publicly released.
This runner uses LLaVA-1.6-NeXT (llava-v1.6-mistral-7b-hf) as a proxy:
- Same LLaVA family, strong on spatial/relation tasks
- Replace model_id when LLaVA-4D is released on HuggingFace

Ref: https://openreview.net/forum?id=URpbmVEsqB
Proxy: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
"""
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor

try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextForConditionalGeneration = None


class LLaVA4DRunner:
    """
    Runner for LLaVA-4D (proxy: LLaVA-1.6-NeXT until release).
    For role-based MAS: Direct / relation-focused perception.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        **kwargs,
    ):
        if LlavaNextForConditionalGeneration is None:
            raise ImportError(
                "LLaVA-4D (proxy) requires transformers with LlavaNext. "
                "pip install transformers>=4.45"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device
        self.model_id = model_id
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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
        **kwargs,
    ) -> str:
        """Generate answer from image + prompt."""
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
        return answer.strip()
