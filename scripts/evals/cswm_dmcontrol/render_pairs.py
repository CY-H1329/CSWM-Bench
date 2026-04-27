#!/usr/bin/env python3
"""
Render GT rollouts for CSWM DMControl action pairs and compute divergence metrics.

Outputs under:
  results/runs/cswm_dmcontrol/<timestamp>/
    videos/<id>_A.mp4
    videos/<id>_Aprime.mp4
    summary.json
    details.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "cswm_dmcontrol" / "pairs.jsonl"


def _parse_env(s: str) -> Tuple[str, str]:
    d, t = s.split(":", 1)
    return d, t


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _rollout(env, actions: List[List[float]], height: int, width: int) -> Tuple[List[np.ndarray], np.ndarray]:
    ts = env.reset()
    frames: List[np.ndarray] = []
    # collect a simple state vector for divergence: qpos+qvel
    def state_vec():
        try:
            qpos = np.asarray(env.physics.data.qpos, dtype=np.float32).ravel()
            qvel = np.asarray(env.physics.data.qvel, dtype=np.float32).ravel()
            return np.concatenate([qpos, qvel], axis=0)
        except Exception:
            return np.zeros((1,), dtype=np.float32)

    states = [state_vec()]
    frames.append(env.physics.render(height=height, width=width, camera_id=0))
    for a in actions:
        ts = env.step(np.asarray(a, dtype=np.float32))
        frames.append(env.physics.render(height=height, width=width, camera_id=0))
        states.append(state_vec())
        if getattr(ts, "last", lambda: False)():
            break
    return frames, np.stack(states, axis=0)


def _save_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole:swingup")
    ap.add_argument("--pairs", default=str(DATA_PATH))
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_pairs", type=int, default=20)
    args = ap.parse_args()

    from dm_control import suite  # type: ignore

    pairs = _read_jsonl(Path(args.pairs))
    pairs = [p for p in pairs if p.get("env") == args.env]
    pairs = pairs[: min(args.max_pairs, len(pairs))]

    out_root = ROOT / "results" / "runs" / "cswm_dmcontrol" / datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir = out_root / "videos"
    out_root.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    details = []
    dists = []

    domain, task = _parse_env(args.env)
    for p in tqdm(pairs, desc="render_gt"):
        env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": int(p["seed"])})
        frames_a, states_a = _rollout(env, p["a"][: args.horizon], args.height, args.width)
        env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": int(p["seed"])})
        frames_b, states_b = _rollout(env, p["a_prime"][: args.horizon], args.height, args.width)

        # Divergence metric: L2 distance between final state vectors (qpos+qvel)
        sa = states_a[-1]
        sb = states_b[-1]
        dist = float(np.linalg.norm(sa - sb))
        dists.append(dist)

        vid_a = (videos_dir / f"{p['id']}_A.mp4").relative_to(ROOT).as_posix()
        vid_b = (videos_dir / f"{p['id']}_Aprime.mp4").relative_to(ROOT).as_posix()
        _save_mp4(frames_a, ROOT / vid_a, fps=args.fps)
        _save_mp4(frames_b, ROOT / vid_b, fps=args.fps)

        details.append(
            {
                "id": p["id"],
                "env": p["env"],
                "seed": p["seed"],
                "k": p["k"],
                "delta": p["delta"],
                "state_l2_final": dist,
                "video_A": vid_a,
                "video_Aprime": vid_b,
            }
        )

    arr = np.asarray(dists, dtype=np.float32) if dists else np.zeros((0,), dtype=np.float32)
    summary = {
        "env": args.env,
        "n": int(len(details)),
        "state_l2_mean": float(arr.mean()) if len(arr) else 0.0,
        "state_l2_median": float(np.median(arr)) if len(arr) else 0.0,
        "state_l2_p90": float(np.quantile(arr, 0.9)) if len(arr) else 0.0,
        "out_dir": out_root.relative_to(ROOT).as_posix(),
    }

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_root / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_root}")


if __name__ == "__main__":
    main()

