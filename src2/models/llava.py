"""
LLaVA inference for STVQA-7K.
Supports LLaVA-1.5 and LLaVA-1.6 (NeXT, e.g. llava-v1.6-mistral-7b-hf).
"""
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor

# 1.6/NeXT uses LlavaNextForConditionalGeneration; 1.5 uses LlavaForConditionalGeneration
try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextForConditionalGeneration = None
from transformers import LlavaForConditionalGeneration


def _is_llava_next(model_id: str) -> bool:
    return "1.6" in model_id or "v1.6" in model_id or "mistral" in model_id.lower() or "next" in model_id.lower()


class LLaVARunner:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.is_next = _is_llava_next(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        load_kwargs = dict(
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            **kwargs,
        )
        # No device_map — run without accelerate
        if self.is_next and LlavaNextForConditionalGeneration is not None:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
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
        if self.is_next:
            # LLaVA-NeXT: conversation + apply_chat_template, then processor(image, prompt)
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
                inputs = self.processor(image, prompt_str, return_tensors="pt").to(self.model.device)
            except TypeError:
                inputs = self.processor(
                    images=[image], text=[prompt_str], padding=True, return_tensors="pt"
                ).to(self.model.device)
        else:
            # LLaVA 1.5 style
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
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
        # decode only the generated part
        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            start = inputs.input_ids.shape[1]
        else:
            start = 0
        answer = self.processor.decode(out[0][start:], skip_special_tokens=True)
        return answer.strip()
