"""
Sa2VA inference (ByteDance/Sa2VA-4B).
Uses model.predict_forward() for image chat.
Requires: transformers, trust_remote_code=True
"""
from typing import Optional
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer


class Sa2VARunner:
    """Runner for Sa2VA (e.g. ByteDance/Sa2VA-4B)."""

    def __init__(
        self,
        model_id: str = "ByteDance/Sa2VA-4B",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        load_kwargs = dict(
            **kwargs,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,  # True causes meta tensor .item() error in InternVisionModel
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            device_map=None,  # avoid accelerate meta-device path
        )
        # Avoid meta tensors: force CPU default during init (InternVisionModel uses .item() on linspace)
        _prev = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
        try:
            self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        finally:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device(_prev)
        if device == "cuda":
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
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
        # Sa2VA format: <image> + text
        text_prompts = f"<image>{prompt}"
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        input_dict = {
            "image": image_rgb,
            "text": text_prompts,
            "past_text": "",
            "mask_prompts": None,
            "tokenizer": self.tokenizer,
        }
        return_dict = self.model.predict_forward(**input_dict)
        return (return_dict.get("prediction") or "").strip()
