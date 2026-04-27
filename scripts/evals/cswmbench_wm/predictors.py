from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict

from envs import (
    DoorState,
    TablePushState,
    door_outcome_at_90,
    marginal_consequence_label,
    divergence_same_or_different,
)


class BasePredictor:
    """
    Minimal WM evaluation interface.

    For each example, return a dict with the same keys as `gt` for that task.
    """

    def predict(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class OraclePredictor(BasePredictor):
    """Uses the ground-truth simulator (upper bound sanity check)."""

    def predict(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        task = ex.get("task")
        if task == "A":
            s1 = DoorState(**ex["case1"]["state"])
            s2 = DoorState(**ex["case2"]["state"])
            out1, reason1 = door_outcome_at_90(s1)
            out2, reason2 = door_outcome_at_90(s2)
            return {
                "divergence": divergence_same_or_different(out1, out2),
                "case1_outcome": out1,
                "case2_outcome": out2,
                # For the toy task, we want the *cause* of divergence.
                "reason_label": "collision_with_swing_arc" if out1 != out2 else (reason1 or reason2),
            }
        if task == "B":
            s = TablePushState(**ex["state"])
            label, reason = marginal_consequence_label(s)
            return {"divergence_label": label, "reason_label": reason}
        return {}


class RandomPredictor(BasePredictor):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def predict(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        task = ex.get("task")
        if task == "A":
            return {
                "divergence": self.rng.choice(["same", "different"]),
                "case1_outcome": self.rng.choice(["hit", "clear"]),
                "case2_outcome": self.rng.choice(["hit", "clear"]),
                "reason_label": self.rng.choice(["collision_with_swing_arc", "outside_swing_arc"]),
            }
        if task == "B":
            return {
                "divergence_label": self.rng.choice(["both_safe", "both_fall", "push2_safe_push6_fall"]),
                "reason_label": self.rng.choice(
                    ["threshold_crossing", "insufficient_displacement", "both_cross_edge_threshold"]
                ),
            }
        return {}


# --- Dreamer adapter (stub) ---


class DreamerPredictor(BasePredictor):
    """
    Stub adaptateur Dreamer.

    À implémenter quand tu as:
      - le code Dreamer (DreamerV2/V3) + checkpoint
      - une fonction qui fait: state + action -> rollout / event probabilities

    Note: pour ces toy envs, on peut entraîner Dreamer très vite (2D déterministe),
    et mesurer si le WM capture bien les seuils et collisions.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        raise NotImplementedError("DreamerPredictor: branchement checkpoint à implémenter.")

    def predict(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

