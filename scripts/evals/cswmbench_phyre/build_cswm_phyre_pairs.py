#!/usr/bin/env python3
"""
Build CSWM-style contrastive pairs from PHYRE.

We search for action pairs (a1, a2) that are:
  - close in action space (small L2 distance)
  - but have different outcomes (success vs fail)

Outputs:
  data/cswmbench_phyre/cswm_phyre_pairs.jsonl
  data/cswmbench_phyre/images/<task_id>.png  (initial scene)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "cswmbench_phyre"
IMG_DIR = OUT_DIR / "images"


def _require_phyre():
    try:
        import phyre  # type: ignore
    except Exception as e:
        raise SystemExit(
            "PHYRE n'est pas installé dans cet environnement.\n"
            "Installe-le (conda/pip) puis relance. Voir: "
            "https://github.com/facebookresearch/phyre/blob/master/INSTALLATION.md\n"
            f"Erreur import: {e}"
        )
    return phyre


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_initial_image(obs: np.ndarray, out_path: Path) -> str:
    """
    obs can be:
      - HxWx3 uint8
      - HxW (int) 'scene' image-like
    We'll normalize to RGB for a viewer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = obs
    if arr.ndim == 2:
        # map int labels to grayscale
        mx = int(arr.max()) if arr.size else 1
        mx = max(mx, 1)
        g = (arr.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
        arr = np.stack([g, g, g], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    im.save(out_path, format="PNG")
    rel = out_path.relative_to(ROOT)
    return rel.as_posix()


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt((d * d).sum()))


def _get_task_ids(phyre, eval_setup: str, split: str, fold_id: int) -> List[str]:
    """
    PHYRE APIs differ slightly across versions. Try common entry points.
    """
    split = split.lower().strip()
    # Newer: phyre.get_fold(eval_setup, fold_id) -> (train_ids, dev_ids, test_ids)
    if hasattr(phyre, "get_fold"):
        folds = phyre.get_fold(eval_setup, fold_id)
        if isinstance(folds, (list, tuple)) and len(folds) >= 3:
            train_ids, dev_ids, test_ids = folds[0], folds[1], folds[2]
            if split in ("train", "tr"):
                return list(train_ids)
            if split in ("dev", "val", "valid", "validation"):
                return list(dev_ids)
            if split in ("test", "te"):
                return list(test_ids)
    # Older fallback: list all task ids in eval setup
    if hasattr(phyre, "get_task_ids_in_eval_setup"):
        all_ids = list(phyre.get_task_ids_in_eval_setup(eval_setup))
        # No split info; return prefix subset for reproducibility
        return all_ids
    raise SystemExit(f"Impossible d'obtenir les tasks pour eval_setup={eval_setup}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_setup", default="ball_cross_template", help="PHYRE eval setup name")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="Which fold split to use")
    ap.add_argument("--fold_id", type=int, default=0)
    ap.add_argument("--max_tasks", type=int, default=100)
    ap.add_argument("--pairs_per_task", type=int, default=3)
    ap.add_argument("--num_actions", type=int, default=2000, help="Candidate actions sampled per task")
    ap.add_argument("--max_action_l2", type=float, default=0.25, help="Keep only pairs closer than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    phyre = _require_phyre()

    _safe_mkdir(IMG_DIR)
    _safe_mkdir(OUT_DIR)

    task_ids = _get_task_ids(phyre, args.eval_setup, args.split, args.fold_id)
    task_ids = task_ids[: min(args.max_tasks, len(task_ids))]

    action_tier = phyre.eval_setup_to_action_tier(args.eval_setup)
    sim = phyre.initialize_simulator(task_ids, action_tier)

    rng = np.random.default_rng(args.seed)

    # Cache initial observations and images
    initial_obs = sim.initial_scenes  # often HxWx? or list

    out_jsonl = OUT_DIR / "cswm_phyre_pairs.jsonl"
    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, task_id in enumerate(tqdm(task_ids, desc="tasks")):
            # Save initial image
            try:
                obs = sim.render(idx) if hasattr(sim, "render") else sim.get_observations([idx])[0]
            except Exception:
                # Fallback: try raw scene
                obs = initial_obs[idx]
            img_rel = _save_initial_image(np.asarray(obs), IMG_DIR / f"{task_id}.png")

            # Sample candidate actions
            # PHYRE provides an action space helper.
            if hasattr(phyre, "get_default_action_space"):
                action_space = phyre.get_default_action_space(action_tier)
                acts = action_space.sample(args.num_actions, rng=rng)
            else:
                # Fallback: uniform random in [0,1]
                dim = 3 if action_tier == "ball" else 6
                acts = rng.random((args.num_actions, dim), dtype=np.float32)

            # Simulate outcomes
            statuses = []
            for a in acts:
                try:
                    r = sim.simulate_action(idx, a, need_images=False)
                    status = int(getattr(r, "status", r[0] if isinstance(r, (list, tuple)) else 0))
                except Exception:
                    status = -1
                statuses.append(status)

            # Map status to success/fail (PHYRE uses phyre.simulation_cache.* constants)
            # We infer success by equality to phyre.SimulationStatus.SOLVED / or numeric 1 in many builds.
            def is_success(st: int) -> bool:
                if hasattr(phyre, "SimulationStatus"):
                    # heuristic: SOLVED exists
                    solved = getattr(phyre.SimulationStatus, "SOLVED", None)
                    if solved is not None and st == int(solved):
                        return True
                # common: 1 means solved
                return st == 1

            succ_idx = [i for i, st in enumerate(statuses) if is_success(st)]
            fail_idx = [i for i, st in enumerate(statuses) if (st != -1 and not is_success(st))]
            if not succ_idx or not fail_idx:
                continue

            # Build close cross-outcome pairs
            pairs = []
            # Sample a subset for efficiency
            rng.shuffle(succ_idx)
            rng.shuffle(fail_idx)
            succ_idx = succ_idx[: min(250, len(succ_idx))]
            fail_idx = fail_idx[: min(250, len(fail_idx))]

            for si in succ_idx:
                a_s = np.asarray(acts[si])
                # find closest failing action
                best = None
                best_d = 1e9
                for fi in fail_idx:
                    d = _l2(a_s, np.asarray(acts[fi]))
                    if d < best_d:
                        best_d = d
                        best = fi
                if best is not None and best_d <= args.max_action_l2:
                    pairs.append((si, best, best_d))
                if len(pairs) >= args.pairs_per_task:
                    break

            for si, fi, dist in pairs:
                rec = {
                    "id": f"cswm_phyre_{task_id}_{si}_{fi}",
                    "benchmark": "cswm_phyre",
                    "task_id": task_id,
                    "eval_setup": args.eval_setup,
                    "action_tier": str(action_tier),
                    "image": img_rel,
                    "a1": [float(x) for x in np.asarray(acts[si]).tolist()],
                    "a2": [float(x) for x in np.asarray(acts[fi]).tolist()],
                    "gt": {
                        "divergence": "different",
                        "a1_outcome": "success",
                        "a2_outcome": "fail",
                        "action_distance_l2": float(dist),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} pairs to {out_jsonl}")
    print(f"Images in {IMG_DIR}")


if __name__ == "__main__":
    main()

