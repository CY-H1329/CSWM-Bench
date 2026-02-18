"""
DeepSeek-VL inference (open-source, GPU).
Requires: transformers with DeepseekVLForConditionalGeneration support.
Model: deepseek-community/deepseek-vl-7b-chat or deepseek-vl-1.3b-chat
"""
import os
import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import torch

try:
    from transformers import DeepseekVLForConditionalGeneration, AutoProcessor
except ImportError:
    DeepseekVLForConditionalGeneration = None
    AutoProcessor = None


class DeepSeekVLRunner:
    """GPU runner for DeepSeek-VL (open-source)."""

    def __init__(
        self,
        model_id: str = "deepseek-community/deepseek-vl-7b-chat",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        if DeepseekVLForConditionalGeneration is None:
            raise ImportError(
                "DeepSeek-VL requires transformers with DeepseekVL support. "
                "Install: pip install transformers>=4.45"
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
                load_kwargs["attn_implementation"] = "sdpa"

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = DeepseekVLForConditionalGeneration.from_pretrained(
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
        max_new_tokens: int = 1024,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        # DeepSeek-VL processor expects "url" in content; use temp file for PIL
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name, format="PNG")
            img_path = f.name
        try:
            url = Path(img_path).as_uri()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": url},
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
        finally:
            try:
                os.unlink(img_path)
            except Exception:
                pass

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        else:
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.model.device, dtype=getattr(self.model, "dtype", None) or torch.bfloat16)

        try:
            from transformers import GenerationConfig
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
            )
            if temperature > 0:
                gen_config.temperature = temperature
            gen_kwargs = {"generation_config": gen_config}
        except ImportError:
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

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
