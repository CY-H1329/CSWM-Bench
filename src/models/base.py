"""
Base interface for VLM runners with role-specific tool integration.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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

    # -------------------------------------------------------------------------
    # Role-specific generation (tool-augmented)
    # -------------------------------------------------------------------------

    def generate_as_direct(self, image: Image.Image, prompt: str, **kwargs) -> str | Dict[str, Any]:
        """Direct 2D: no tools, just image + prompt."""
        return self.generate(image, prompt, **kwargs)

    def generate_as_3d(
        self,
        image: Image.Image,
        prompt: str,
        depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        3D role: Depth Anything V2 → depth map → [image | depth] concat → model.
        """
        try:
            from src.tools.depth import DepthTool
        except ImportError:
            try:
                from tools.depth import DepthTool
            except ImportError:
                raise ImportError(
                    "generate_as_3d requires src.tools. Depth tool uses Depth Anything V2. "
                    "Run from project root. pip install transformers>=4.45"
                ) from None

        device = 0 if getattr(self, "device", None) == "cuda" else -1
        depth_tool = DepthTool(device=device, model_id=depth_model_id)
        enriched = depth_tool.estimate_and_concat(image)
        role_prompt = (
            "[3D Role] The left half is the original image; the right half is the depth map "
            "(brighter = closer). Use depth information for spatial reasoning.\n\n" + prompt
        )
        return self.generate(enriched, role_prompt, **kwargs)

    def generate_as_scene_graph(
        self,
        image: Image.Image,
        prompt: str,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        SceneGraph role: Grounding DINO → objects + relations → graph text → prompt.
        """
        try:
            from src.tools.scene_graph import SceneGraphTool
        except ImportError:
            try:
                from tools.scene_graph import SceneGraphTool
            except ImportError:
                raise ImportError(
                    "generate_as_scene_graph requires src.tools. "
                    "Scene graph uses Grounding DINO. pip install transformers"
                ) from None

        sg_tool = SceneGraphTool(model_id=grounding_model_id)
        graph_text = sg_tool.build_graph(image)
        role_prompt = (
            "[Scene Graph Role] Use the following scene structure for spatial reasoning:\n\n"
            f"{graph_text}\n\n"
            "Question: " + prompt
        )
        return self.generate(image, role_prompt, **kwargs)

    def generate_by_role(
        self,
        image: Image.Image,
        prompt: str,
        role: str,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        Generate according to assigned role (score-based assignment).
        role: "Direct" | "3D" | "SceneGraph"
        """
        role = role or "Direct"
        if role == "3D":
            return self.generate_as_3d(image, prompt, **kwargs)
        if role == "SceneGraph":
            return self.generate_as_scene_graph(image, prompt, **kwargs)
        return self.generate_as_direct(image, prompt, **kwargs)
