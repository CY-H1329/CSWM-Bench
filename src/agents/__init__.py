"""
Multi-Agent System: Head → Perception → Reasoning
- Head-Agent: classifie le task (sans recevoir la catégorie)
- Perception Agent: extrait les infos (reçoit query + task_class)
- Reasoning Agent: raisonne et répond (reçoit query + task_class + perception)
"""
from .pipeline import run_mas_pipeline, MAS_PIPELINE_PROMPTS

__all__ = ["run_mas_pipeline", "MAS_PIPELINE_PROMPTS"]
