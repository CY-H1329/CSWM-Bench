"""
Claude (Anthropic) vision API for MAS.
"""
import os
import base64
import io
from typing import Optional
from PIL import Image

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


def _sanitize_text(s: str) -> str:
    if not s:
        return s
    return s.replace("\u2028", " ").replace("\u2029", " ").strip()


class ClaudeRunner:
    """Runner for Claude vision models via Anthropic API."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        if Anthropic is None:
            raise ImportError("Install anthropic: pip install anthropic")
        key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        key = key.replace("\u2028", "").replace("\u2029", "")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=key)
        self.model_id = model_id

    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=format)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        prompt = _sanitize_text(prompt)
        b64 = self._image_to_base64(image)
        msg = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=max(0.0, temperature),
        )
        text = msg.content[0].text if msg.content else ""
        return text.strip()
