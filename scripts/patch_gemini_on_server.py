#!/usr/bin/env python3
"""
서버에 Gemini 지원이 없을 때 한 번 실행.
프로젝트 루트에서: python scripts/patch_gemini_on_server.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_PY = '''"""
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
    return s.replace("\\u2028", " ").replace("\\u2029", " ").strip()


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
        key = key.replace("\\u2028", "").replace("\\u2029", "")
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
'''

def main():
    # 1) Create gemini.py
    gemini_path = ROOT / "src" / "models" / "gemini.py"
    gemini_path.parent.mkdir(parents=True, exist_ok=True)
    gemini_path.write_text(GEMINI_PY)
    print(f"Created {gemini_path}")

    # 2) Patch run_eval.py
    run_eval_path = ROOT / "run_eval.py"
    text = run_eval_path.read_text()
    if "GeminiRunner" in text and 'model_name == "gemini"' in text:
        print("run_eval.py already has Gemini support.")
        return
    if "from src.models.gemini import GeminiRunner" not in text:
        text = text.replace(
            "from src.models.gpt import GPTRunner\n",
            "from src.models.gpt import GPTRunner\nfrom src.models.gemini import GeminiRunner\n",
        )
    if 'elif model_name == "gemini":' not in text:
        text = text.replace(
            """        runner = GPTRunner(
            model_id=m_cfg.get("model_id", "gpt-4o"),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")""",
            """        runner = GPTRunner(
            model_id=m_cfg.get("model_id", "gpt-4o"),
            api_key=api_key,
        )
    elif model_name == "gemini":
        m_cfg = config.get("models", {}).get("gemini", {})
        if not m_cfg.get("enabled", True):
            return None
        api_key = os.environ.get(m_cfg.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(f"[skip] {model_name}: no GEMINI_API_KEY / GOOGLE_API_KEY")
            return None
        runner = GeminiRunner(
            model_id=m_cfg.get("model_id", "gemini-2.0-flash"),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")""",
        )
    if "model_name in (\"gpt\", \"gemini\")" not in text:
        text = text.replace(
            'if model_name == "gpt":',
            'if model_name in ("gpt", "gemini"):',
        )
    run_eval_path.write_text(text)
    print("Patched run_eval.py for Gemini.")
    print("Run: python run_eval.py --models qwen llava gemini --split val")

if __name__ == "__main__":
    main()
