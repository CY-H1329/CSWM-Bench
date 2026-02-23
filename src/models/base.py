from abc import ABC, abstractmethod
from typing import List, Any
from PIL import Image


class BaseVLM(ABC):
    """Base interface for VLM runners (Qwen, LLaVA, GPT)."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Return model answer text given image and text prompt."""
        pass

    def run_batch(self, images: List[Image.Image], prompts: List[str], **kwargs) -> List[str]:
        """Run on multiple (image, prompt) pairs. Default: sequential."""
        return [self.generate(img, p, **kwargs) for img, p in zip(images, prompts)]
