"""
Qwen3-VL inference (Qwen3-VL-4B-Instruct).
Requires: transformers>=4.51 (Qwen3VLForConditionalGeneration)
"""
from typing import Optional
from PIL import Image
import torch

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None


class Qwen3Runner:
    """Runner for Qwen3-VL (e.g. Qwen3-VL-4B-Instruct)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3-VL requires transformers>=4.51. "
                "Install: pip install transformers>=4.51"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device

        load_kwargs = dict(
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass  # fallback to default attention

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
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
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        # Qwen3-VL does not support top_k/top_p; omit to avoid generation flags warning
        for k, v in kwargs.items():
            if k not in ("top_k", "top_p") and v is not None:
                gen_kwargs[k] = v
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs.get("input_ids", inputs)
        in_len = input_ids.shape[1] if hasattr(input_ids, "shape") and len(input_ids.shape) > 1 else len(input_ids)
        generated_trimmed = [generated_ids[0][in_len:]] if hasattr(generated_ids, "shape") else [generated_ids[in_len:]]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""
