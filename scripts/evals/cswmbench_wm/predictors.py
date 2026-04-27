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
            if out1 == "hit" and out2 == "clear":
                reason = "case1_collision"
            elif out2 == "hit" and out1 == "clear":
                reason = "case2_collision"
            else:
                reason = "outside_swing_arc"
            return {
                "divergence": divergence_same_or_different(out1, out2),
                "case1_outcome": out1,
                "case2_outcome": out2,
                "reason_label": reason,
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
                "reason_label": self.rng.choice(["case1_collision", "case2_collision", "outside_swing_arc"]),
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


class LearnedMLPPredictor(BasePredictor):
    """
    Baseline appris (supervisé) sur états vectoriels.
    Utilise le checkpoint produit par `train_baselines.py`.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.device = device
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.labels = ckpt.get("labels", {})

        # Minimal re-definition of the same architecture (avoid cross-file imports)
        def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        hidden = 128
        self.a_backbone = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.b_backbone = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

        self.a_div = nn.Linear(hidden, len(self.labels.get("A_div", ["same", "different"])))
        self.a_o1 = nn.Linear(hidden, len(self.labels.get("A_outcome", ["hit", "clear"])))
        self.a_o2 = nn.Linear(hidden, len(self.labels.get("A_outcome", ["hit", "clear"])))
        self.a_reason = nn.Linear(hidden, len(self.labels.get("A_reason", ["case1_collision", "case2_collision", "outside_swing_arc"])))
        self.b_label = nn.Linear(hidden, len(self.labels.get("B_label", ["both_safe", "both_fall", "push2_safe_push6_fall"])))
        self.b_reason = nn.Linear(hidden, len(self.labels.get("B_reason", ["threshold_crossing", "insufficient_displacement", "both_cross_edge_threshold"])))

        self._modules = [
            self.a_backbone,
            self.b_backbone,
            self.a_div,
            self.a_o1,
            self.a_o2,
            self.a_reason,
            self.b_label,
            self.b_reason,
        ]
        for m in self._modules:
            m.to(self.device)
            m.eval()

        # Load parameters (keys align with MultiHeadModel in train_baselines.py)
        state = ckpt["state_dict"]

        def _load(mod, prefix: str):
            sub = {k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")}
            mod.load_state_dict(sub, strict=True)

        _load(self.a_backbone, "a_backbone")
        _load(self.b_backbone, "b_backbone")
        _load(self.a_div, "a_div")
        _load(self.a_o1, "a_o1")
        _load(self.a_o2, "a_o2")
        _load(self.a_reason, "a_reason")
        _load(self.b_label, "b_label")
        _load(self.b_reason, "b_reason")

    def _encode_a(self, ex: Dict[str, Any]):
        s1 = ex["case1"]["state"]
        s2 = ex["case2"]["state"]
        r = float(s1["door_radius"])
        x1, y1 = map(float, s1["box_center"])
        w1, h1 = map(float, s1["box_size"])
        x2, y2 = map(float, s2["box_center"])
        w2, h2 = map(float, s2["box_size"])
        return [r, x1, y1, w1, h1, x2, y2, w2, h2]

    def _encode_b(self, ex: Dict[str, Any]):
        s = ex["state"]
        return [float(s["edge_distance"]), float(s.get("object_radius", 1.0))]

    def predict(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if ex.get("task") == "A":
            x = torch.tensor(self._encode_a(ex), dtype=torch.float32, device=self.device).unsqueeze(0)
            h = self.a_backbone(x)
            div = int(self.a_div(h).argmax(dim=-1).item())
            o1 = int(self.a_o1(h).argmax(dim=-1).item())
            o2 = int(self.a_o2(h).argmax(dim=-1).item())
            rs = int(self.a_reason(h).argmax(dim=-1).item())
            return {
                "divergence": self.labels["A_div"][div],
                "case1_outcome": self.labels["A_outcome"][o1],
                "case2_outcome": self.labels["A_outcome"][o2],
                "reason_label": self.labels["A_reason"][rs],
            }

        x = torch.tensor(self._encode_b(ex), dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.b_backbone(x)
        lb = int(self.b_label(h).argmax(dim=-1).item())
        rs = int(self.b_reason(h).argmax(dim=-1).item())
        return {
            "divergence_label": self.labels["B_label"][lb],
            "reason_label": self.labels["B_reason"][rs],
        }

