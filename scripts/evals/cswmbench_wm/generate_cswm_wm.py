#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from envs import (
    DoorState,
    TablePushState,
    door_outcome_at_90,
    marginal_consequence_label,
    divergence_same_or_different,
)


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "cswmbench_wm"


def _mk_task_a(rng: random.Random, i: int) -> Dict:
    # Create two states that differ only by box center.
    r = 1.0
    box_size = (0.35, 0.35)

    # Randomize which case is the "hit" one to make reason_label meaningful.
    hit_is_case1 = bool(rng.getrandbits(1))

    def sample_hit_state() -> DoorState:
        return DoorState(
            door_radius=r,
            box_center=(rng.uniform(0.25, 0.55), rng.uniform(0.25, 0.55)),
            box_size=box_size,
        )

    def sample_clear_state() -> DoorState:
        return DoorState(
            door_radius=r,
            box_center=(rng.uniform(0.95, 1.35), rng.uniform(0.95, 1.35)),
            box_size=box_size,
        )

    if hit_is_case1:
        s1 = sample_hit_state()
        s2 = sample_clear_state()
    else:
        s1 = sample_clear_state()
        s2 = sample_hit_state()

    out1, _ = door_outcome_at_90(s1)
    out2, _ = door_outcome_at_90(s2)

    # We want divergence by construction; resample until holds.
    tries = 0
    while out1 == out2 and tries < 50:
        # Resample the "clear" state if needed.
        if hit_is_case1:
            s2 = sample_clear_state()
        else:
            s1 = sample_clear_state()
        out1, _ = door_outcome_at_90(s1)
        out2, _ = door_outcome_at_90(s2)
        tries += 1

    gt = {
        "divergence": divergence_same_or_different(out1, out2),
        "case1_outcome": out1,
        "case2_outcome": out2,
        "reason_label": "case1_collision" if hit_is_case1 else "case2_collision",
    }
    return {
        "id": f"cswm_wm_A_{i:04d}",
        "task": "A",
        "category": "geometric_clearance",
        "case1": {"state": asdict(s1), "action": {"open_degrees": 90}},
        "case2": {"state": asdict(s2), "action": {"open_degrees": 90}},
        "gt": gt,
    }


def _mk_task_b(rng: random.Random, i: int) -> Dict:
    # Build three bands like the image version, but with explicit edge clearance.
    bands = [
        ("both_safe", (6.5, 9.0)),
        ("both_fall", (0.8, 1.8)),
        ("push2_safe_push6_fall", (2.5, 5.5)),
    ]
    tgt, (lo, hi) = bands[i % len(bands)]
    edge_distance = rng.uniform(lo, hi)
    s = TablePushState(edge_distance=edge_distance, object_radius=1.0)
    label, reason = marginal_consequence_label(s)
    assert label == tgt

    # Pair of actions: push2 vs push6
    a1 = {"push": 2.0}
    a2 = {"push": 6.0}
    gt = {
        "divergence": divergence_same_or_different("fall" if label in ("both_fall",) else "safe", "fall" if label in ("both_safe",) else "safe"),
        "divergence_label": label,
        "reason_label": reason,
    }
    return {
        "id": f"cswm_wm_B_{i:04d}",
        "task": "B",
        "category": "marginal_consequence",
        "state": asdict(s),
        "actions": {"a1": a1, "a2": a2},
        "gt": gt,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cswmbench_wm.jsonl"
    rng = random.Random(42)

    items: List[Dict] = []
    for i in range(64):
        items.append(_mk_task_a(rng, i))
    for i in range(64):
        items.append(_mk_task_b(rng, i))
    rng.shuffle(items)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()

