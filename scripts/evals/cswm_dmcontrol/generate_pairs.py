#!/usr/bin/env python3
"""
Generate CSWM-style contrastive action pairs for DMControl.

We store minimal metadata (env, seed, action sequences A/A', perturb index, delta).
Rendering is done separately in render_pairs.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "cswm_dmcontrol"


def _parse_env(s: str) -> Tuple[str, str]:
    if ":" not in s:
        raise SystemExit("--env must be like domain:task, e.g. cartpole:swingup")
    d, t = s.split(":", 1)
    return d, t


@dataclass(frozen=True)
class Pair:
    id: str
    env: str
    seed: int
    horizon: int
    action_dim: int
    k: int
    delta: float
    a: List[List[float]]
    a_prime: List[List[float]]


def _sample_actions(rng: random.Random, horizon: int, action_dim: int, scale: float = 1.0) -> List[List[float]]:
    # DMControl actions usually in [-1, 1]
    return [[rng.uniform(-scale, scale) for _ in range(action_dim)] for _ in range(horizon)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole:swingup")
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.15, help="Perturbation magnitude (L_inf on one action component).")
    args = ap.parse_args()

    # We import dm_control only to discover action_dim reliably.
    domain, task = _parse_env(args.env)
    from dm_control import suite  # type: ignore

    env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": args.seed})
    action_spec = env.action_spec()
    action_dim = int(getattr(action_spec, "shape", (0,))[0] or 0)
    if action_dim <= 0:
        raise SystemExit("Could not infer action_dim from dm_control env.action_spec()")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pairs.jsonl"

    rng = random.Random(args.seed)
    pairs: List[Pair] = []
    for i in range(args.n_pairs):
        seed_i = rng.randint(0, 10_000_000)
        r2 = random.Random(seed_i)
        a = _sample_actions(r2, args.horizon, action_dim, scale=1.0)
        a_p = [row[:] for row in a]
        k = r2.randrange(0, args.horizon)
        j = r2.randrange(0, action_dim)
        # Small delta on one component; clamp to [-1,1]
        a_p[k][j] = max(-1.0, min(1.0, a_p[k][j] + (args.delta if r2.random() < 0.5 else -args.delta)))
        pairs.append(
            Pair(
                id=f"cswm_dm_{domain}_{task}_{i:04d}",
                env=args.env,
                seed=seed_i,
                horizon=args.horizon,
                action_dim=action_dim,
                k=k,
                delta=float(args.delta),
                a=a,
                a_prime=a_p,
            )
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()

