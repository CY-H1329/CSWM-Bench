from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# -------------------------
# Task A (Door clearance)
# -------------------------


@dataclass(frozen=True)
class DoorState:
    # Door hinge at origin; door sweeps quarter circle (0→90°).
    door_radius: float  # meters (or arbitrary units)
    # Axis-aligned box center position in hinge frame (x right, y up)
    box_center: Tuple[float, float]
    box_size: Tuple[float, float]  # (w, h)


def door_outcome_at_90(state: DoorState) -> Tuple[str, str]:
    """
    Outcome labels:
      - hit: door sweep intersects box
      - clear: no intersection

    Reason labels (coarse, for single-state outcome):
      - collision_with_swing_arc
      - outside_swing_arc
    """
    # Conservative check: if any point of the box is within radius band in 1st quadrant.
    (cx, cy) = state.box_center
    (bw, bh) = state.box_size
    # Box corners
    xs = [cx - bw / 2, cx + bw / 2]
    ys = [cy - bh / 2, cy + bh / 2]
    r = state.door_radius

    def in_first_quadrant(x: float, y: float) -> bool:
        return x >= 0 and y >= 0

    hit = False
    for x in xs:
        for y in ys:
            if not in_first_quadrant(x, y):
                continue
            if (x * x + y * y) <= (r * r):
                hit = True
                break
        if hit:
            break

    if hit:
        return "hit", "collision_with_swing_arc"
    return "clear", "outside_swing_arc"


# -------------------------
# Task B (Marginal push)
# -------------------------


@dataclass(frozen=True)
class TablePushState:
    # Distance from object's rightmost extent to table edge
    edge_distance: float  # same unit as pushes
    object_radius: float


def table_push_outcome(state: TablePushState, push: float) -> bool:
    """Return True if the object falls off after pushing toward the edge."""
    # Falls if displacement exceeds available clearance.
    # Use edge_distance measured from object's rightmost point to edge.
    return push >= state.edge_distance


def marginal_consequence_label(state: TablePushState, push2: float = 2.0, push6: float = 6.0) -> Tuple[str, str]:
    fall2 = table_push_outcome(state, push2)
    fall6 = table_push_outcome(state, push6)
    if (not fall2) and (not fall6):
        return "both_safe", "insufficient_displacement"
    if fall2 and fall6:
        return "both_fall", "both_cross_edge_threshold"
    return "push2_safe_push6_fall", "threshold_crossing"


def divergence_same_or_different(out1: str, out2: str) -> str:
    return "same" if out1 == out2 else "different"


def to_dict(obj) -> Dict:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(type(obj))

