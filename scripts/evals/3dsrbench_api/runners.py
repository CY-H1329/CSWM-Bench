"""
API runners for 3DSRBench — Claude, GPT-4o, DeepSeek-VL, Gemini.
Separate from src/models to avoid modifying existing code.
"""
import os
import io
import base64
import time
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
    def __init__(self, model_id: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None):
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
    """GPT-4o / GPT-5.x via OpenAI API. Retries on rate limits (상용 실행용)."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("Install: pip install openai")
        key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.model_id = model_id
        self.max_retries = max_retries if max_retries is not None else int(
            os.environ.get("OPENAI_MAX_RETRIES", "8")
        )

    def _is_retryable(self, err: BaseException) -> bool:
        msg = str(err).lower()
        if "429" in msg or "rate" in msg or "timeout" in msg or "connection" in msg:
            return True
        try:
            import openai

            types = tuple(
                t
                for name in ("RateLimitError", "APIConnectionError", "APITimeoutError")
                for t in (getattr(openai, name, None),)
                if t is not None
            )
            return isinstance(err, types) if types else False
        except Exception:
            return False

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
        mid = (self.model_id or "").lower()
        tok_param = {"max_completion_tokens": max_tokens} if "gpt-5" in mid else {"max_tokens": max_tokens}
        req = dict(
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
            **tok_param,
        )
        if not mid.startswith("gpt-5") and not mid.startswith("o1") and not mid.startswith("o3"):
            req["temperature"] = max(0.0, temperature)

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**req)
                return (resp.choices[0].message.content or "").strip()
            except BaseException as e:
                last_err = e
                if attempt + 1 >= self.max_retries or not self._is_retryable(e):
                    raise
                wait = min(8.0 * (2**attempt), 120.0)
                time.sleep(wait)
        raise last_err  # pragma: no cover


class DeepSeekVLRunner:
    """DeepSeek-VL : OpenRouter (image_url) ou api.deepseek.com (/v1/vision)."""
    def __init__(
        self,
        model_id: str = "deepseek-vl",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
    ):
        key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set DEEPSEEK_API_KEY")
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        # OpenRouter accepte image_url ; api.deepseek.com non → on utilise /v1/vision
        self._use_openrouter = "openrouter" in base_url.lower()

        if self._use_openrouter:
            if not OPENAI_AVAILABLE:
                raise ImportError("Install: pip install openai")
            self.client = OpenAI(api_key=key, base_url=self.base_url)
        else:
            try:
                import requests
                self._requests = requests
            except ImportError:
                raise ImportError("Install: pip install requests")

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

        if self._use_openrouter:
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

        # api.deepseek.com : endpoint /v1/vision (chat/completions rejette image_url)
        url = f"{self.base_url}/v1/vision"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "image": b64,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": max(0.0, temperature),
        }
        r = self._requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "choices" in data and data["choices"]:
            return (data["choices"][0].get("message", {}).get("content") or "").strip()
        if "text" in data:
            return (data["text"] or "").strip()
        return ""


class OpenRouterRunner:
    """Generic OpenRouter runner (OpenAI-compatible API). GLM-5 is text-only (no vision)."""
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        text_only: bool = False,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("Install: pip install openai")
        key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_id = model_id
        self.text_only = text_only

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        prompt = _sanitize(prompt)
        if self.text_only:
            content = [{"type": "text", "text": prompt}]
        else:
            b64 = _img_to_base64(image)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
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
