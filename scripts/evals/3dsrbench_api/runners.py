"""
API runners for 3DSRBench — Claude, GPT-4o, DeepSeek-VL, Gemini.
Separate from src/models to avoid modifying existing code.
"""
import os
import io
import base64
from typing import Optional
from PIL import Image

# Claude (Anthropic)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# GPT-4o, DeepSeek (OpenAI-compatible)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Gemini
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def _img_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _sanitize(s: str) -> str:
    return (s or "").replace("\u2028", " ").replace("\u2029", " ").strip()


class ClaudeRunner:
    """Claude 3.5 Sonnet vision via Anthropic API."""
    def __init__(self, model_id: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Install: pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self.model_id = model_id

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        prompt = _sanitize(prompt)
        b64 = _img_to_base64(image)
        msg = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=max(0.0, temperature),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return (msg.content[0].text if msg.content else "").strip()


class GPT4oRunner:
    """GPT-4o via OpenAI API (reuses GPTRunner pattern)."""
    def __init__(self, model_id: str = "gpt-4o", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("Install: pip install openai")
        key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.model_id = model_id

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        prompt = _sanitize(prompt)
        b64 = _img_to_base64(image)
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=max(0.0, temperature),
        )
        return (resp.choices[0].message.content or "").strip()


class DeepSeekVLRunner:
    """DeepSeek-VL via OpenAI-compatible API."""
    def __init__(
        self,
        model_id: str = "deepseek-vl",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("Install: pip install openai")
        key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_id = model_id

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        prompt = _sanitize(prompt)
        b64 = _img_to_base64(image)
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=max(0.0, temperature),
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiRunner:
    """Gemini (Gemini Robotics-ER or gemini-2.0-flash) via Google GenAI."""
    def __init__(self, model_id: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        if not GENAI_AVAILABLE:
            raise ImportError("Install: pip install google-genai")
        key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model_id = model_id

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        prompt = _sanitize(prompt)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[img_part, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=max(0.0, temperature),
            ),
        )
        return (resp.text or "").strip()
