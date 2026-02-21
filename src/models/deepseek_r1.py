"""
DeepSeek-R1 Runner pour Spatial MAS — Reasoning Agent.

Rôle MAS: Reasoning Agent (Step 3).
Text-only: reçoit les sorties concaténées des agents (réponses, rôles, poids, z_i)
et produit la réponse finale.

F(x, I, {(a_i, k_i, w^{i,k_i}, z_i)}) — le prompt contient tout le contexte.
Quand l'image I est nécessaire, passer la description dans le prompt ou utiliser DeepSeek-VL.

Refs:
- https://arxiv.org/html/2205.12880v2 (Collaborative Beam Search, multi-LLM consensus)
- https://arxiv.org/html/2503.04954v1 (Trust-based MAS)
- https://huggingface.co/deepseek-ai/DeepSeek-R1

Supports: local (transformers) ou API (openrouter/deepseek).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

logger = logging.getLogger(__name__)

# HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-R1
DEEPSEEK_R1_HF_URL = "https://huggingface.co/deepseek-ai/DeepSeek-R1"


class DeepSeekR1Runner:
    """
    Runner pour DeepSeek-R1 (Reasoning Agent).

    HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-R1

    Text-only: reçoit les outputs des agents en texte, produit la réponse finale.
    Step 3 MAS: F(x, I, {(a_i, k_i, w^{i,k_i}, z_i)}).
    """

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1",
        device: Optional[str] = None,
        api_key_env: Optional[str] = "DEEPSEEK_API_KEY",
        use_api: bool = False,
        torch_dtype: Optional["torch.dtype"] = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.use_api = use_api
        self.api_key = (
            os.environ.get(api_key_env or "", "").strip() if api_key_env else ""
        )
        self.hf_url = DEEPSEEK_R1_HF_URL

        if use_api and self.api_key:
            self.model = None
            self.tokenizer = None
            self.device = None
            return

        if not _HF_AVAILABLE:
            raise ImportError(
                "DeepSeek-R1 local requiert transformers, torch. "
                "pip install transformers torch"
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else None
        dtype = torch_dtype or (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()
        self.device = device

    def generate(
        self,
        image: Optional[object] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_k: int = 0,
        top_p: float = 0.0,
        return_full_output: bool = False,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        Génère le reasoning output. prompt contient les outputs des agents.

        Signature compatible BaseVLM et pipeline MAS:
        - generate(image, prompt) : image ignoré (text-only)
        - generate(prompt) : prompt seul

        Args:
            image: Ignoré (text-only); passer le contexte image dans le prompt si besoin
            prompt: Texte complet (query + agent outputs + rôles + poids + z_i)
            max_new_tokens: 1024 par défaut pour reasoning long

        Returns:
            str ou dict si return_full_output
        """
        if prompt is None and image is not None and isinstance(image, str):
            prompt, image = image, None
        elif prompt is None:
            prompt = kwargs.pop("prompt", "") or ""
        if not prompt:
            return ""
        if self.use_api and self.api_key:
            answer = self._generate_api(prompt, max_new_tokens, temperature)
            if return_full_output:
                return {"answer": answer, "raw": answer, "final_answer": answer}
            return answer

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=pad_id,
            temperature=temperature if temperature > 0 else None,
            **kwargs,
        )

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        generated = out[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        if return_full_output:
            return {
                "answer": answer,
                "raw": answer,
                "final_answer": self._extract_final_answer(answer),
            }
        return answer

    def _generate_api(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Utilise l'API DeepSeek (text completion)."""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("DeepSeek API error: %s", e)
            return f"[API Error: {e}]"

    def _extract_final_answer(self, text: str) -> str:
        """Extrait 'Final Answer:' pour le pipeline MAS."""
        for marker in ["Final Answer:", "Final answer:"]:
            if marker in text:
                idx = text.find(marker) + len(marker)
                return text[idx:].strip().split("\n")[0][:500]
        return text[:500]

    def run_batch(
        self,
        prompts: List[str],
        images: Optional[List[object]] = None,
        **kwargs,
    ) -> List[str]:
        """Exécute une série de prompts (text-only). images ignoré."""
        return [self.generate(p, **kwargs) for p in prompts]
