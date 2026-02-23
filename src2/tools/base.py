"""
Base interface for MAS v2 tools.

All tools take PIL Image and return a string summary suitable for
injection into specialist prompts.
"""
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseTool(ABC):
    """Base class for spatial reasoning tools."""

    @abstractmethod
    def run(self, image: Image.Image) -> str:
        """Process image and return a textual summary for prompt injection."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for logging."""
        pass
