"""
Spatial MAS Runners — Ré-export des runners depuis src.models.
"""
import sys
from pathlib import Path

# Ajouter le projet au path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models import get_runner, list_agents
from src.models.qwen3 import Qwen3Runner
from src.models.sa2va import Sa2VARunner
from src.models.llava4d import LLaVA4DRunner
from src.models.deepseek_r1 import DeepSeekR1Runner

try:
    from src.models.spatialrgpt import SpatialRGPTRunner
except ImportError:
    SpatialRGPTRunner = None

try:
    from src.models.spatialreasoner import SpatialReasonerRunner
except ImportError:
    SpatialReasonerRunner = None

__all__ = [
    "get_runner",
    "list_agents",
    "Qwen3Runner",
    "Sa2VARunner",
    "LLaVA4DRunner",
    "SpatialRGPTRunner",
    "SpatialReasonerRunner",
    "DeepSeekR1Runner",
]
