"""
Registry of agent runners for MAS.

Maps agent names (config keys) to runner classes and default model IDs.
Use get_runner(agent_name, **kwargs) to instantiate.
"""
from typing import Any, Dict, Optional

from .base import BaseVLM
from .qwen3 import Qwen3Runner
from .sa2va import Sa2VARunner
from .gpt import GPTRunner
from .gemini import GeminiRunner
from .deepseek_r1 import DeepSeekR1Runner
from .deepseek_vl import DeepSeekVLRunner

try:
    from .llava4d import LLaVA4DRunner
except ImportError:
    LLaVA4DRunner = None
try:
    from .claude import ClaudeRunner
except ImportError:
    ClaudeRunner = None

# Optional runners (may require extra setup)
try:
    from .spatialrgpt import SpatialRGPTRunner
except ImportError:
    SpatialRGPTRunner = None
try:
    from .spatialreasoner import SpatialReasonerRunner
except ImportError:
    SpatialReasonerRunner = None

# Agent name -> (RunnerClass, default_model_id, is_vision)
AGENT_REGISTRY: Dict[str, tuple] = {
    # Perception agents (image + prompt -> answer)
    "qwen3_4b": (Qwen3Runner, "Qwen/Qwen3-VL-4B-Instruct", True),
    "sa2va": (Sa2VARunner, "ByteDance/Sa2VA-4B", True),
    "llava4d": (LLaVA4DRunner, "llava-hf/llava-v1.6-mistral-7b-hf", True)
    if LLaVA4DRunner
    else (None, None, True),
    "spatialrgpt": (SpatialRGPTRunner, "a8cheng/SpatialRGPT-VILA1.5-8B", True)
    if SpatialRGPTRunner
    else (None, None, True),
    "spatialreasoner": (SpatialReasonerRunner, "ccvl/SpatialReasoner", True)
    if SpatialReasonerRunner
    else (None, None, True),
    # API-based
    "gpt4o": (GPTRunner, "gpt-4o", True),
    "gemini_robotics_er": (GeminiRunner, "gemini-2.0-flash", True),
    "claude_sonnet_4_5": (ClaudeRunner, "claude-sonnet-4-20250514", True)
    if ClaudeRunner
    else (None, None, True),
    # Reasoning (text-only)
    "deepseek_r1": (DeepSeekR1Runner, "deepseek-ai/DeepSeek-R1", False),
    # Vision reasoning (image + text)
    "deepseek_vl": (DeepSeekVLRunner, "deepseek-ai/deepseek-vl-7b-chat", True),
}


def get_runner(
    agent_name: str,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> Optional[BaseVLM]:
    """
    Instantiate a runner for the given agent name.

    Args:
        agent_name: Config key (e.g. qwen3_4b, sa2va, llava4d, spatialrgpt, spatialreasoner)
        model_id: Override default model ID
        device: cuda/cpu
        **kwargs: Passed to runner __init__

    Returns:
        Runner instance or None if agent not supported
    """
    entry = AGENT_REGISTRY.get(agent_name)
    if not entry:
        return None
    cls, default_id, _ = entry
    if cls is None:
        return None
    mid = model_id or default_id
    if not mid:
        return None
    init_kw = dict(**kwargs)
    if device is not None:
        init_kw["device"] = device
    return cls(model_id=mid, **init_kw)


# HuggingFace URLs per agent (model_id -> HF page)
AGENT_HF_URLS = {
    "qwen3_4b": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
    "sa2va": "https://huggingface.co/ByteDance/Sa2VA-4B",
    "llava4d": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf",
    "spatialrgpt": "https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B",
    "spatialreasoner": "https://huggingface.co/ccvl/SpatialReasoner",
    "deepseek_r1": "https://huggingface.co/deepseek-ai/DeepSeek-R1",
    "deepseek_vl": "https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat",
}


def list_agents() -> Dict[str, Dict[str, Any]]:
    """Return agent name -> {runner, model_id, is_vision, hf_url}."""
    out = {}
    for name, entry in AGENT_REGISTRY.items():
        cls, mid, is_vision = entry if isinstance(entry, tuple) else (None, None, True)
        out[name] = {
            "runner": cls.__name__ if cls else None,
            "model_id": mid,
            "is_vision": is_vision,
            "hf_url": AGENT_HF_URLS.get(name),
        }
    return out
