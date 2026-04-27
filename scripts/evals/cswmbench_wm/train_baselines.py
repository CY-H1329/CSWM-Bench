#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _one_hot(index: int, n: int) -> List[float]:
    v = [0.0] * n
    v[index] = 1.0
    return v


LABELS = {
    "A_reason": ["case1_collision", "case2_collision", "outside_swing_arc"],
    "A_outcome": ["hit", "clear"],
    "A_div": ["same", "different"],
    "B_label": ["both_safe", "both_fall", "push2_safe_push6_fall"],
    "B_reason": ["threshold_crossing", "insufficient_displacement", "both_cross_edge_threshold"],
}


def _encode_task_a(ex: Dict[str, Any]) -> Tuple[List[float], Dict[str, int]]:
    s1 = ex["case1"]["state"]
    s2 = ex["case2"]["state"]
    # Features: [r, box1(x,y,w,h), box2(x,y,w,h)]
    r = float(s1["door_radius"])
    x1, y1 = map(float, s1["box_center"])
    w1, h1 = map(float, s1["box_size"])
    x2, y2 = map(float, s2["box_center"])
    w2, h2 = map(float, s2["box_size"])
    feats = [r, x1, y1, w1, h1, x2, y2, w2, h2]

    gt = ex["gt"]
    y = {
        "div": LABELS["A_div"].index(gt["divergence"]),
        "o1": LABELS["A_outcome"].index(gt["case1_outcome"]),
        "o2": LABELS["A_outcome"].index(gt["case2_outcome"]),
        "reason": LABELS["A_reason"].index(gt["reason_label"]),
    }
    return feats, y


def _encode_task_b(ex: Dict[str, Any]) -> Tuple[List[float], Dict[str, int]]:
    s = ex["state"]
    feats = [float(s["edge_distance"]), float(s.get("object_radius", 1.0))]
    gt = ex["gt"]
    y = {
        "label": LABELS["B_label"].index(gt["divergence_label"]),
        "reason": LABELS["B_reason"].index(gt["reason_label"]),
    }
    return feats, y


class CSWMDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, List[float], Dict[str, int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        task, x, y = self.rows[idx]
        return task, torch.tensor(x, dtype=torch.float32), y


def _collate(batch):
    tasks = [b[0] for b in batch]
    # All samples in a batch must have identical feature length.
    xs = torch.stack([b[1] for b in batch], dim=0)
    ys = [b[2] for b in batch]
    return tasks, xs, ys


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadModel(nn.Module):
    def __init__(self, a_in: int = 9, b_in: int = 2, hidden: int = 128):
        super().__init__()
        self.a_backbone = nn.Sequential(nn.Linear(a_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.b_backbone = nn.Sequential(nn.Linear(b_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

        # Register heads as real submodules (do NOT wrap in a dataclass).
        self.a_div = nn.Linear(hidden, len(LABELS["A_div"]))
        self.a_o1 = nn.Linear(hidden, len(LABELS["A_outcome"]))
        self.a_o2 = nn.Linear(hidden, len(LABELS["A_outcome"]))
        self.a_reason = nn.Linear(hidden, len(LABELS["A_reason"]))
        self.b_label = nn.Linear(hidden, len(LABELS["B_label"]))
        self.b_reason = nn.Linear(hidden, len(LABELS["B_reason"]))

    def forward(self, task: str, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if task == "A":
            h = self.a_backbone(x)
            return {
                "div": self.a_div(h),
                "o1": self.a_o1(h),
                "o2": self.a_o2(h),
                "reason": self.a_reason(h),
            }
        else:
            h = self.b_backbone(x)
            return {
                "label": self.b_label(h),
                "reason": self.b_reason(h),
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "cswmbench_wm" / "cswmbench_wm.jsonl"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs" / "cswmbench_wm_models"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data = _read_jsonl(Path(args.data))
    rows_a = []
    rows_b = []
    for ex in data:
        if ex["task"] == "A":
            x, y = _encode_task_a(ex)
            rows_a.append(("A", x, y))
        elif ex["task"] == "B":
            x, y = _encode_task_b(ex)
            rows_b.append(("B", x, y))

    rng.shuffle(rows_a)
    rng.shuffle(rows_b)

    def split(rows):
        n = len(rows)
        n_train = int(0.8 * n)
        return rows[:n_train], rows[n_train:]

    train_a, test_a = split(rows_a)
    train_b, test_b = split(rows_b)

    train_dl_a = DataLoader(CSWMDataset(train_a), batch_size=64, shuffle=True, collate_fn=_collate)
    train_dl_b = DataLoader(CSWMDataset(train_b), batch_size=64, shuffle=True, collate_fn=_collate)
    test_dl_a = DataLoader(CSWMDataset(test_a), batch_size=128, shuffle=False, collate_fn=_collate)
    test_dl_b = DataLoader(CSWMDataset(test_b), batch_size=128, shuffle=False, collate_fn=_collate)

    model = MultiHeadModel().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    def eval_split(dl_a, dl_b):
        model.eval()
        stats = {"A": {"div": 0, "o1": 0, "o2": 0, "reason": 0, "n": 0}, "B": {"label": 0, "reason": 0, "n": 0}}
        with torch.no_grad():
            for tasks, xs, ys in dl_a:
                xs = xs.to(args.device)
                for i, t in enumerate(tasks):
                    out = model(t, xs[i : i + 1]).copy()
                    if t == "A":
                        stats["A"]["n"] += 1
                        for k in ("div", "o1", "o2", "reason"):
                            pred = int(out[k].argmax(dim=-1).item())
                            if pred == ys[i][k]:
                                stats["A"][k] += 1
            for tasks, xs, ys in dl_b:
                xs = xs.to(args.device)
                for i, t in enumerate(tasks):
                    out = model(t, xs[i : i + 1]).copy()
                    stats["B"]["n"] += 1
                    for k in ("label", "reason"):
                        pred = int(out[k].argmax(dim=-1).item())
                        if pred == ys[i][k]:
                            stats["B"][k] += 1
        return stats

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        # Alternate batches from A and B to avoid mixed feature dims.
        for (tasks, xs, ys) in train_dl_a:
            xs = xs.to(args.device)
            loss = 0.0
            for i, t in enumerate(tasks):
                out = model("A", xs[i : i + 1])
                loss = loss + ce(out["div"], torch.tensor([ys[i]["div"]], device=args.device))
                loss = loss + ce(out["o1"], torch.tensor([ys[i]["o1"]], device=args.device))
                loss = loss + ce(out["o2"], torch.tensor([ys[i]["o2"]], device=args.device))
                loss = loss + ce(out["reason"], torch.tensor([ys[i]["reason"]], device=args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        for (tasks, xs, ys) in train_dl_b:
            xs = xs.to(args.device)
            loss = 0.0
            for i, t in enumerate(tasks):
                out = model("B", xs[i : i + 1])
                loss = loss + ce(out["label"], torch.tensor([ys[i]["label"]], device=args.device))
                loss = loss + ce(out["reason"], torch.tensor([ys[i]["reason"]], device=args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if (ep + 1) % 25 == 0 or ep == args.epochs - 1:
            st = eval_split(test_dl_a, test_dl_b)
            # simple scalar for model selection
            a = st["A"]
            b = st["B"]
            score = 0.0
            if a["n"]:
                score += (a["div"] + a["reason"] + a["o1"] + a["o2"]) / (4.0 * a["n"])
            if b["n"]:
                score += (b["label"] + b["reason"]) / (2.0 * b["n"])
            score /= 2.0
            if score > best:
                best = score
                out_dir = Path(args.out) / "mlp_multihead"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "labels": LABELS,
                    },
                    out_dir / "ckpt.pt",
                )
            print(f"epoch={ep+1} best={best:.3f} testA(n={a['n']}) div={a['div']/max(1,a['n']):.3f} reason={a['reason']/max(1,a['n']):.3f} "
                  f"testB(n={b['n']}) label={b['label']/max(1,b['n']):.3f} reason={b['reason']/max(1,b['n']):.3f}")


if __name__ == "__main__":
    main()

