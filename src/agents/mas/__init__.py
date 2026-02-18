"""
Spatial Multi-Agent System (MAS) with:
- Head-Agent (GPT-5.2, fixed): category inference, agent selection, coordination policy
- 3 Specialist Agents: autonomous strategy, CoT + answer
- Reasoning Agent (DeepSeek-VL): final answer, justification, score update
"""
from .config import AGENT_PROFILES, load_agent_profiles, SCORE_DELTA_CORRECT, SCORE_DELTA_WRONG, INITIAL_WEIGHT
from .pipeline import run_spatial_mas_pipeline
from .score_manager import ScoreManager

__all__ = [
    "run_spatial_mas_pipeline",
    "ScoreManager",
    "AGENT_PROFILES",
    "load_agent_profiles",
    "SCORE_DELTA_CORRECT",
    "SCORE_DELTA_WRONG",
    "INITIAL_WEIGHT",
]
