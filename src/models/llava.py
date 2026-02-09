"""
LLaVA inference for STVQA-7K.
"""
from typing import Optional
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor


class LLaVARunner:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # H100 등 GPU: device_map="auto" 로 가용 GPU 사용
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
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
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> str:
        # LLaVA 1.5 style prompt
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                **kwargs,
            )
        answer = self.processor.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return answer.strip()
