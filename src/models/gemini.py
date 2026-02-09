"""
Gemini (Google) vision API for STVQA-7K.
"""
import os
import io
from typing import Optional
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def _sanitize_text(s: str) -> str:
    if not s:
        return s
    return s.replace("\u2028", " ").replace("\u2029", " ").strip()


class GeminiRunner:
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        if genai is None or types is None:
            raise ImportError("Install: pip install google-genai")
        key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        key = key.replace("\u2028", "").replace("\u2029", "")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model_id = model_id

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        prompt = _sanitize_text(prompt)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=max(0.0, temperature),
            ),
        )
        text = (response.text or "").strip()
        return text
