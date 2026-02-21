"""
DeepSeek-R1 inference for Reasoning Agent.

Text-only reasoning model. Receives concatenated agent outputs (answers, roles, weights, z_i)
and produces final answer. For multi-modal reasoning with image, use DeepSeek-VL instead.

Supports: local (transformers) or API (openrouter/deepseek).
Ref: https://huggingface.co/deepseek-ai/DeepSeek-R1
"""
import os
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class DeepSeekR1Runner:
    """
    Runner for DeepSeek-R1 (Reasoning Agent).
    Text-only: receives agent outputs as text, produces final answer.
    For role-based MAS Step 3: F(x, I, {(a_i, k_i, w^{i,k_i}, z_i)}).
    When image I is needed, pass image description in prompt or use DeepSeek-VL.
    """

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1",
        device: Optional[str] = None,
        api_key_env: Optional[str] = "DEEPSEEK_API_KEY",
        use_api: bool = False,
        **kwargs,
    ):
        self.model_id = model_id
        self.use_api = use_api
        self.api_key = os.environ.get(api_key_env or "", "").strip() if api_key_env else ""

        if use_api and self.api_key:
            self.model = None
            self.tokenizer = None
            self.device = None
            return

        if not _HF_AVAILABLE:
            raise ImportError("DeepSeek-R1 local requires transformers, torch. pip install transformers torch")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()
        self.device = device

    def generate(
        self,
        prompt: str,
        image: Optional[object] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate reasoning output. prompt contains agent outputs.
        image: ignored for text-only; pass image context in prompt if needed.
        """
        if self.use_api and self.api_key:
            return self._generate_api(prompt, max_new_tokens, temperature)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with __import__("torch").inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=pad_id,
                temperature=temperature if temperature > 0 else None,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_api(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Use DeepSeek API (text completion)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[API Error: {e}]"
