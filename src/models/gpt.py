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
        raw_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        # U+2028/U+2029 등이 API 키에 들어가면 헤더 인코딩 에러 발생
        api_key_clean = raw_key.replace("\u2028", "").replace("\u2029", "").strip()
        self.client = OpenAI(api_key=api_key_clean)
        self.model_id = model_id

    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=format)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _sanitize_text(s: str) -> str:
        """ASCII 인코딩 오류 방지: U+2028, U+2029 등 제거."""
        if not s:
            return s
        return s.replace("\u2028", " ").replace("\u2029", " ").strip()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        prompt = self._sanitize_text(prompt)
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
