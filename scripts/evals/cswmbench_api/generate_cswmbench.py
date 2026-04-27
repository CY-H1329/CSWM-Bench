#!/usr/bin/env python3
"""
CSWM-Bench (toy) generator — Tasks A/B with synthetic diagram images.

Goal: give you an executable contrastive benchmark ASAP (for meeting evidence).
This produces:
  - data/cswmbench/cswmbench.jsonl
  - data/cswmbench/images/*.png

Tasks
  A: Geometric Clearance (contrastive pair of scenes; open door 90°)
  B: Marginal Consequence (single scene; push distance threshold)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "cswmbench"
IMG_DIR = OUT_DIR / "images"


def _ensure_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def _try_font(size: int = 16) -> ImageFont.ImageFont:
    # Portable fallback: PIL default bitmap font (always available)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_caption(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    font = _try_font(16)
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _save(im: Image.Image, rel_path: Path) -> str:
    path = (ROOT / rel_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")
    return str(rel_path.as_posix())


def _mk_canvas(w: int = 512, h: int = 512, bg: Tuple[int, int, int] = (18, 18, 22)) -> Image.Image:
    return Image.new("RGB", (w, h), bg)


def _task_a_scene(box_inside_arc: bool, seed: int) -> Tuple[Image.Image, Dict]:
    """
    Draw a top-down door swing setup.
    We keep most geometry identical; only box position changes.

    Outcome:
      - hit_box if box intersects the swing arc at 90°
      - clear otherwise
    """
    rng = random.Random(seed)
    im = _mk_canvas()
    draw = ImageDraw.Draw(im)

    # Coordinate system: origin top-left
    hinge = (140, 260)
    door_len = 220
    door_thickness = 26

    # Draw walls (simple L shape around hinge)
    draw.rectangle([hinge[0] - 10, hinge[1] - 200, hinge[0] + 10, hinge[1] + 200], fill=(60, 60, 70))
    draw.rectangle([hinge[0] - 200, hinge[1] - 10, hinge[0] + 200, hinge[1] + 10], fill=(60, 60, 70))

    # Door closed: extends to the right from hinge
    x0, y0 = hinge
    door_closed = [x0, y0 - door_thickness // 2, x0 + door_len, y0 + door_thickness // 2]
    draw.rectangle(door_closed, fill=(160, 160, 170))
    _draw_caption(draw, (16, 16), "Task A: Door clearance (open to 90°)")

    # Draw swing arc (quarter circle)
    arc_bbox = [x0 - door_len, y0 - door_len, x0 + door_len, y0 + door_len]
    draw.arc(arc_bbox, start=270, end=360, fill=(120, 200, 255), width=4)
    # Draw a dashed-ish radius line at 90° open (upwards)
    for t in range(0, door_len, 12):
        draw.line([(x0, y0), (x0, y0 - t)], fill=(120, 200, 255), width=1)

    # Box: only change is whether it's inside the arc region near the sweep
    # We'll place it near the top-right quadrant relative to hinge.
    if box_inside_arc:
        bx = x0 + rng.randint(55, 95)
        by = y0 - rng.randint(60, 120)
        gt_outcome = "hit"
        gt_reason = "collision_with_swing_arc"
    else:
        bx = x0 + rng.randint(150, 190)
        by = y0 - rng.randint(140, 180)
        gt_outcome = "clear"
        gt_reason = "outside_swing_arc"

    box_w = rng.randint(46, 62)
    box_h = rng.randint(46, 62)
    draw.rectangle([bx, by, bx + box_w, by + box_h], fill=(230, 150, 80))
    draw.rectangle([bx, by, bx + box_w, by + box_h], outline=(0, 0, 0), width=2)
    _draw_caption(draw, (bx, by - 18), "BOX")

    # Mark hinge
    draw.ellipse([x0 - 6, y0 - 6, x0 + 6, y0 + 6], fill=(255, 80, 80))

    meta = {
        "hinge": hinge,
        "door_len": door_len,
        "box_inside_arc": box_inside_arc,
        "outcome": gt_outcome,
        "reason_label": gt_reason,
    }
    return im, meta


def _task_b_scene(edge_cm: float, seed: int) -> Tuple[Image.Image, Dict]:
    """
    Draw a simple top-down table edge and a cup with distance-to-edge in cm.
    Two actions: push 2cm vs push 6cm (to the right, toward edge).

    GT label among:
      - both_safe
      - both_fall
      - push2_safe_push6_fall
    """
    rng = random.Random(seed)
    im = _mk_canvas()
    draw = ImageDraw.Draw(im)
    _draw_caption(draw, (16, 16), "Task B: Marginal consequence (push 2cm vs 6cm)")

    # Table (top view). Right border is the falling edge.
    table = [70, 120, 450, 420]
    draw.rectangle(table, fill=(90, 110, 90))
    draw.rectangle(table, outline=(0, 0, 0), width=3)
    edge_x = table[2]
    _draw_caption(draw, (edge_x - 90, table[1] - 22), "EDGE →")

    # Cup as circle, placed edge_cm from edge.
    # Map cm to pixels (roughly)
    px_per_cm = 14
    cup_r = 26
    dist_px = int(round(edge_cm * px_per_cm))
    cx = edge_x - dist_px - cup_r
    cy = rng.randint(table[1] + 80, table[3] - 80)
    draw.ellipse([cx - cup_r, cy - cup_r, cx + cup_r, cy + cup_r], fill=(220, 220, 235), outline=(0, 0, 0), width=2)
    _draw_caption(draw, (cx - 18, cy - 56), f"{edge_cm:.1f}cm")

    # Push arrows: 2cm and 6cm
    for push_cm, color, yoff in [(2.0, (255, 210, 120), -48), (6.0, (255, 140, 120), 48)]:
        dx = int(round(push_cm * px_per_cm))
        x1, y1 = cx, cy + yoff
        x2, y2 = cx + dx, cy + yoff
        draw.line([(x1, y1), (x2, y2)], fill=color, width=5)
        draw.polygon([(x2, y2), (x2 - 12, y2 - 7), (x2 - 12, y2 + 7)], fill=color)
        _draw_caption(draw, (x1 - 8, y1 - 18), f"push {push_cm:.0f}cm")

    # Simple physics rule: fall if pushed distance >= edge distance.
    # edge distance is measured from cup center to edge minus radius -> approx.
    # We encode GT with a conservative threshold at edge_cm.
    push2 = 2.0
    push6 = 6.0
    fall2 = push2 >= edge_cm
    fall6 = push6 >= edge_cm
    if (not fall2) and (not fall6):
        gt = "both_safe"
        reason = "insufficient_displacement"
    elif fall2 and fall6:
        gt = "both_fall"
        reason = "both_cross_edge_threshold"
    else:
        gt = "push2_safe_push6_fall"
        reason = "threshold_crossing"

    meta = {"edge_cm": edge_cm, "gt": gt, "reason_label": reason}
    return im, meta


@dataclass(frozen=True)
class Item:
    id: str
    task: str
    images: List[str]
    prompt: str
    gt: Dict
    category: str


def build_items(n_a: int = 24, n_b: int = 24, seed: int = 42) -> List[Item]:
    rng = random.Random(seed)
    items: List[Item] = []

    # Task A: pairs
    for i in range(n_a):
        base_seed = rng.randint(0, 10_000_000)
        im1, m1 = _task_a_scene(box_inside_arc=True, seed=base_seed)
        im2, m2 = _task_a_scene(box_inside_arc=False, seed=base_seed)  # same seed → near-identical styling

        rel1 = Path("data/cswmbench/images") / f"A_{i:03d}_case1.png"
        rel2 = Path("data/cswmbench/images") / f"A_{i:03d}_case2.png"
        p1 = _save(im1, rel1)
        p2 = _save(im2, rel2)

        gt = {
            "divergence": "different",
            "case1_outcome": m1["outcome"],  # hit
            "case2_outcome": m2["outcome"],  # clear
            "reason_label": "collision_with_swing_arc",
        }
        prompt = (
            "You are shown TWO images (case1, case2). In both cases, a door is opened to 90 degrees.\n"
            "Question: Do the outcomes differ between case1 and case2?\n\n"
            "Choose divergence: [same, different]\n"
            "Outcome labels per case: [hit, clear]\n"
            "Reason labels: [collision_with_swing_arc, outside_swing_arc]\n\n"
            "Return ONLY strict JSON with keys: divergence, case1_outcome, case2_outcome, reason_label."
        )
        items.append(
            Item(
                id=f"cswm_A_{i:03d}",
                task="A",
                images=[p1, p2],
                prompt=prompt,
                gt=gt,
                category="geometric_clearance",
            )
        )

    # Task B: single image
    # We sample edge distances to produce all three GT classes.
    # - edge_cm in (6.5..9.0) => both_safe for pushes 2/6
    # - edge_cm in (0.5..1.8) => both_fall
    # - edge_cm in (2.5..5.5) => threshold crossing (2 safe, 6 fall)
    bands = [
        ("both_safe", (6.5, 9.0)),
        ("both_fall", (0.8, 1.8)),
        ("push2_safe_push6_fall", (2.5, 5.5)),
    ]
    for i in range(n_b):
        base_seed = rng.randint(0, 10_000_000)
        tgt, (lo, hi) = bands[i % len(bands)]
        edge_cm = rng.uniform(lo, hi)
        im, meta = _task_b_scene(edge_cm=edge_cm, seed=base_seed)
        assert meta["gt"] == tgt
        rel = Path("data/cswmbench/images") / f"B_{i:03d}.png"
        p = _save(im, rel)

        gt = {"divergence_label": meta["gt"], "reason_label": meta["reason_label"], "edge_cm": edge_cm}
        prompt = (
            "You are shown ONE image. A cup is on a table near the right edge.\n"
            "Two actions are considered: push 2cm to the right vs push 6cm to the right.\n\n"
            "Choose ONE label:\n"
            "- both_safe\n"
            "- both_fall\n"
            "- push2_safe_push6_fall\n\n"
            "Reason labels: [threshold_crossing, insufficient_displacement, both_cross_edge_threshold]\n\n"
            "Return ONLY strict JSON with keys: divergence_label, reason_label."
        )
        items.append(
            Item(
                id=f"cswm_B_{i:03d}",
                task="B",
                images=[p],
                prompt=prompt,
                gt=gt,
                category="marginal_consequence",
            )
        )

    rng.shuffle(items)
    return items


def main() -> None:
    _ensure_dirs()
    items = build_items()
    out_path = OUT_DIR / "cswmbench.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "id": it.id,
                "task": it.task,
                "category": it.category,
                "images": it.images,
                "prompt": it.prompt,
                "gt": it.gt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {out_path}")
    print(f"Images in {IMG_DIR}")


if __name__ == "__main__":
    main()

