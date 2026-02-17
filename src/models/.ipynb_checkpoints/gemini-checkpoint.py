"""
Gemini (Google) vision API for STVQA-7K.
- 사용처: Google AI Studio API (API 키). Google Cloud 크레딧은 Vertex AI에서만 사용 가능.
- 429: 무료 한도(분당/하루 요청) 초과. 재시도+대기 적용.
"""
import os
import io
import time
from typing import Optional
from PIL import Image

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError:
    genai = None
    types = None
    ClientError = Exception


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
        retry_429: int = 5,
        delay_between_requests: float = 6.0,
        **kwargs,
    ) -> str:
        prompt = _sanitize_text(prompt)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        last_err = None
        for attempt in range(max(1, retry_429)):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=max(0.0, temperature),
                    ),
                )
                return (response.text or "").strip()
            except Exception as e:
                last_err = e
                is_429 = getattr(e, "status_code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_429 and attempt < retry_429 - 1:
                    wait = 60 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise
        if last_err is not None:
            raise last_err
        return ""
