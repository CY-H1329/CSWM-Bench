"""
GPT-4o / GPT-4V evaluation via OpenAI API for STVQA-7K.
"""
import os
import base64
import io
from typing import Optional
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class GPTRunner:
    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        if OpenAI is None:
            raise ImportError("Install openai: pip install openai")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
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
        b64 = self._image_to_base64(image)
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=max(0.0, temperature),
        )
        return (resp.choices[0].message.content or "").strip()
