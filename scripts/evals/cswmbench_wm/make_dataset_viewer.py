#!/usr/bin/env python3
"""
Create an HTML viewer for CSWM WM-track (state/action).

Outputs:
  reports/cswmbench_wm_viewer/index.html

This helps you *visually* explain the task in slides (even though the WM-track
is state/action based). We render simple diagrams from the param states.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _try_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _caption(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    font = _try_font(15)
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _mk_canvas(w: int = 520, h: int = 360) -> Image.Image:
    return Image.new("RGB", (w, h), (18, 18, 22))


def _render_task_a(case: Dict[str, Any], title: str) -> Image.Image:
    """
    Render door swing + box in hinge coordinates.
    """
    st = case["state"]
    r = float(st["door_radius"])
    (cx, cy) = map(float, st["box_center"])
    (bw, bh) = map(float, st["box_size"])

    im = _mk_canvas()
    d = ImageDraw.Draw(im)
    _caption(d, (14, 12), title)

    # Map hinge frame (meters) to pixels
    origin = (120, 270)
    px = 160  # px per unit

    def to_px(x: float, y: float) -> Tuple[int, int]:
        # hinge frame: x right, y up
        return (int(origin[0] + x * px), int(origin[1] - y * px))

    # Draw axes
    d.line([origin, (origin[0] + 300, origin[1])], fill=(80, 80, 95), width=2)
    d.line([origin, (origin[0], origin[1] - 240)], fill=(80, 80, 95), width=2)
    _caption(d, (origin[0] + 305, origin[1] - 10), "x")
    _caption(d, (origin[0] - 10, origin[1] - 260), "y")

    # Draw arc (0->90deg)
    rr = int(r * px)
    bbox = [origin[0] - rr, origin[1] - rr, origin[0] + rr, origin[1] + rr]
    d.arc(bbox, start=270, end=360, fill=(120, 200, 255), width=4)

    # Draw closed door (along +x)
    door_len = rr
    d.rectangle([origin[0], origin[1] - 10, origin[0] + door_len, origin[1] + 10], fill=(165, 165, 175))
    d.ellipse([origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5], fill=(255, 80, 80))

    # Box
    x0, y0 = to_px(cx - bw / 2, cy - bh / 2)
    x1, y1 = to_px(cx + bw / 2, cy + bh / 2)
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    d.rectangle([left, top, right, bottom], fill=(230, 150, 80), outline=(0, 0, 0), width=2)
    _caption(d, (left, top - 18), "BOX")
    return im


def _render_task_b(ex: Dict[str, Any]) -> Image.Image:
    st = ex["state"]
    edge = float(st["edge_distance"])
    im = _mk_canvas()
    d = ImageDraw.Draw(im)
    _caption(d, (14, 12), "Task B (push 2 vs 6)")

    # Simple table with right edge
    table = [60, 70, 500, 320]
    d.rectangle(table, fill=(90, 110, 90), outline=(0, 0, 0), width=3)
    _caption(d, (table[2] - 90, table[1] - 22), "EDGE →")

    # Place object based on edge distance (arbitrary scale)
    px_per_unit = 35
    r = 20
    cx = int(table[2] - edge * px_per_unit - r)
    cy = int((table[1] + table[3]) / 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 220, 235), outline=(0, 0, 0), width=2)
    _caption(d, (cx - 30, cy - 52), f"edge={edge:.2f}")

    # Push arrows
    for push, color, yoff in [(2.0, (255, 210, 120), -50), (6.0, (255, 140, 120), 50)]:
        dx = int(push * px_per_unit)
        x1, y1 = cx, cy + yoff
        x2, y2 = cx + dx, cy + yoff
        d.line([(x1, y1), (x2, y2)], fill=color, width=5)
        d.polygon([(x2, y2), (x2 - 12, y2 - 7), (x2 - 12, y2 + 7)], fill=color)
        _caption(d, (x1 - 8, y1 - 18), f"push {push:.0f}")

    return im


def main() -> None:
    data_path = ROOT / "data" / "cswmbench_wm" / "cswmbench_wm.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}. Run generate_cswm_wm.py first.")

    items = _read_jsonl(data_path)
    out_dir = ROOT / "reports" / "cswmbench_wm_viewer"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    cards = []
    for ex in items:
        ex_id = ex.get("id", "")
        task = ex.get("task", "")
        cat = ex.get("category", "")
        gt = ex.get("gt", {}) or {}

        if task == "A":
            im1 = _render_task_a(ex["case1"], "case1 (open 90°)")
            im2 = _render_task_a(ex["case2"], "case2 (open 90°)")
            p1 = img_dir / f"{ex_id}_case1.png"
            p2 = img_dir / f"{ex_id}_case2.png"
            im1.save(p1)
            im2.save(p2)
            imgs_html = (
                '<div class="imggrid">'
                f'<div><div class="label">case1</div><img src="images/{html.escape(p1.name)}"/></div>'
                f'<div><div class="label">case2</div><img src="images/{html.escape(p2.name)}"/></div>'
                "</div>"
            )
        else:
            im = _render_task_b(ex)
            p = img_dir / f"{ex_id}.png"
            im.save(p)
            imgs_html = f'<div class="imggrid one"><img src="images/{html.escape(p.name)}"/></div>'

        gt_pre = html.escape(json.dumps(gt, ensure_ascii=False, indent=2))
        ex_pre = html.escape(json.dumps(ex, ensure_ascii=False, indent=2)[:4000])
        cards.append(
            f"""
            <div class="card" data-task="{html.escape(str(task))}" data-cat="{html.escape(str(cat))}">
              <div class="top">
                <div>
                  <div class="title">{html.escape(str(ex_id))}</div>
                  <div class="meta">task: <b>{html.escape(str(task))}</b> · category: <b>{html.escape(str(cat))}</b></div>
                </div>
              </div>
              {imgs_html}
              <details>
                <summary>GT + raw example</summary>
                <div class="twocol">
                  <div><div class="label">gt</div><pre>{gt_pre}</pre></div>
                  <div><div class="label">example</div><pre>{ex_pre}</pre></div>
                </div>
              </details>
            </div>
            """
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CSWM WM-track Viewer</title>
  <style>
    :root {{
      --bg:#0b0b0f;--panel:#111117;--text:#e9e9f2;--muted:#a8a8b6;--border:#242430;--accent:#8ad1ff;
    }}
    body {{ background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui; margin:0; padding:24px; }}
    .header {{ display:flex;gap:16px;align-items:flex-end;justify-content:space-between;margin-bottom:18px; }}
    h1 {{ margin:0;font-size:18px; }}
    .muted {{ color:var(--muted);font-size:13px; }}
    .filters {{ display:flex; gap:10px; align-items:center; }}
    select {{ background:var(--panel);color:var(--text);border:1px solid var(--border);padding:8px 10px;border-radius:10px;font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }}
    .card {{ background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px; }}
    .title {{ font-weight:700;font-size:14px; }}
    .meta {{ color:var(--muted);font-size:12px; }}
    .imggrid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:10px 0 8px; }}
    .imggrid.one {{ grid-template-columns:1fr; }}
    img {{ width:100%; max-width:520px; border-radius:10px; border:1px solid #2a2a2a; }}
    .label {{ color:var(--muted);font-size:12px;margin-bottom:6px; }}
    details {{ border-top:1px dashed var(--border); margin-top:10px; padding-top:10px; }}
    summary {{ cursor:pointer;color:var(--accent);font-size:13px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#0d0d13; border:1px solid var(--border); padding:10px; border-radius:10px; font-size:12px; line-height:1.35; }}
    .twocol {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }}
    @media (max-width:860px) {{ .twocol {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>CSWM WM-track Viewer</h1>
      <div class="muted">{len(items)} samples · file: {html.escape(str(data_path.relative_to(ROOT)))}</div>
    </div>
    <div class="filters">
      <div class="muted">Filter</div>
      <select id="taskSel">
        <option value="">all tasks</option>
        <option value="A">Task A</option>
        <option value="B">Task B</option>
      </select>
      <select id="catSel">
        <option value="">all categories</option>
        <option value="geometric_clearance">geometric_clearance</option>
        <option value="marginal_consequence">marginal_consequence</option>
      </select>
    </div>
  </div>
  <div class="grid" id="grid">{''.join(cards)}</div>
  <script>
    const taskSel = document.getElementById('taskSel');
    const catSel = document.getElementById('catSel');
    const cards = Array.from(document.querySelectorAll('.card'));
    function apply() {{
      const t = taskSel.value;
      const c = catSel.value;
      for (const el of cards) {{
        const okT = !t || el.dataset.task === t;
        const okC = !c || el.dataset.cat === c;
        el.style.display = (okT && okC) ? 'block' : 'none';
      }}
    }}
    taskSel.addEventListener('change', apply);
    catSel.addEventListener('change', apply);
  </script>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote viewer: {out_path}")


if __name__ == "__main__":
    main()

