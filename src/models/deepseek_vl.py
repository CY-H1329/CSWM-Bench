"""
DeepSeek-VL inference (open-source, GPU).
Uses pipeline for PIL image support.
Model: deepseek-community/deepseek-vl-7b-chat or deepseek-vl-1.3b-chat
"""
import io
import base64
from typing import Optional
from PIL import Image
import torch

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


def _normalize_image_for_deepseek_vl(image: Image.Image, target: int = 1024) -> Image.Image:
    """
    DeepSeek-VL hybrid uses SAM for high-res patches; it requires exact spatial size (e.g. 1024×1024).
    CV-Bench images can be 1023×1023 or other sizes → ValueError in patch_embed without resize.
    """
    img = image.convert("RGB")
    if img.size == (target, target):
        return img
    return img.resize((target, target), Image.Resampling.LANCZOS)


class DeepSeekVLRunner:
    """GPU runner for DeepSeek-VL (open-source)."""

    def __init__(
        self,
        model_id: str = "deepseek-community/deepseek-vl-7b-chat",
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        if pipeline is None:
            raise ImportError("DeepSeek-VL requires transformers. pip install transformers>=4.45")
        device = device or (0 if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        # Use float32 to match processor output (pixel_values); avoids SAM vision encoder dtype mismatch
        self.pipe = pipeline(
            task="image-text-to-text",
            model=model_id,
            device=device,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **kwargs,
        )
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
        image = _normalize_image_for_deepseek_vl(image)
        # Pipeline: try PIL in content first, else data URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            out = self.pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
        except (ValueError, TypeError, KeyError):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            messages[0]["content"][0] = {"type": "image", "url": f"data:image/png;base64,{b64}"}
            out = self.pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
        if out and isinstance(out, list) and len(out) > 0:
            text = out[0]
            if isinstance(text, dict):
                return (text.get("generated_text") or text.get("text") or "").strip()
            return str(text).strip()
        return ""
