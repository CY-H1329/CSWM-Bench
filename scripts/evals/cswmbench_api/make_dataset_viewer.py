#!/usr/bin/env python3
"""
Create a lightweight HTML viewer for CSWM-Bench (image track).

Outputs:
  reports/cswmbench_api_viewer/index.html

It lets you visually inspect:
  - the contrastive pair (Task A) or single image (Task B)
  - prompt + GT labels (divergence/outcomes/reason)
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _copy_image(rel_path: str, images_dir: Path) -> str:
    """
    Make viewer portable: copy images into <viewer>/images/ and reference them locally.
    Returns a relative path like "images/<name>.png".
    """
    src = (ROOT / rel_path).resolve()
    name = Path(rel_path).name
    dst = images_dir / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return f"images/{name}"


def _img_tag(local_rel: str, max_w: int = 420) -> str:
    return (
        f'<img src="{html.escape(local_rel)}" '
        f'style="max-width:{max_w}px;width:100%;border-radius:10px;border:1px solid #2a2a2a;" />'
    )


def main() -> None:
    data_path = ROOT / "data" / "cswmbench" / "cswmbench.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}. Run generate_cswmbench.py first.")

    items = _read_jsonl(data_path)
    out_dir = ROOT / "reports" / "cswmbench_api_viewer"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    cards = []
    for ex in items:
        ex_id = ex.get("id", "")
        task = ex.get("task", "")
        category = ex.get("category", "")
        imgs = ex.get("images", []) or []
        gt = ex.get("gt", {}) or {}
        prompt = ex.get("prompt", "")

        if len(imgs) == 2:
            p1 = _copy_image(imgs[0], images_dir)
            p2 = _copy_image(imgs[1], images_dir)
            imgs_html = (
                '<div class="imggrid">'
                f'<div><div class="label">case1</div>{_img_tag(p1)}</div>'
                f'<div><div class="label">case2</div>{_img_tag(p2)}</div>'
                "</div>"
            )
        elif len(imgs) == 1:
            p = _copy_image(imgs[0], images_dir)
            imgs_html = f'<div class="imggrid one">{_img_tag(p, max_w=520)}</div>'
        else:
            imgs_html = '<div class="muted">No images</div>'

        gt_pre = html.escape(json.dumps(gt, ensure_ascii=False, indent=2))
        prompt_pre = html.escape(prompt)
        cards.append(
            f"""
            <div class="card" data-task="{html.escape(str(task))}" data-cat="{html.escape(str(category))}">
              <div class="top">
                <div>
                  <div class="title">{html.escape(str(ex_id))}</div>
                  <div class="meta">task: <b>{html.escape(str(task))}</b> · category: <b>{html.escape(str(category))}</b></div>
                </div>
              </div>
              {imgs_html}
              <details>
                <summary>Prompt + GT</summary>
                <div class="twocol">
                  <div>
                    <div class="label">prompt</div>
                    <pre>{prompt_pre}</pre>
                  </div>
                  <div>
                    <div class="label">gt</div>
                    <pre>{gt_pre}</pre>
                  </div>
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
  <title>CSWM-Bench Viewer (image track)</title>
  <style>
    :root {{
      --bg: #0b0b0f;
      --panel: #111117;
      --text: #e9e9f2;
      --muted: #a8a8b6;
      --border: #242430;
      --accent: #8ad1ff;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      margin: 0;
      padding: 24px;
    }}
    .header {{
      display:flex;gap:16px;align-items:flex-end;justify-content:space-between;
      margin-bottom: 18px;
    }}
    h1 {{ margin:0; font-size: 18px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .filters {{ display:flex; gap:10px; align-items:center; }}
    select {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 10px;
      border-radius: 10px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }}
    .top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 8px; }}
    .title {{ font-weight: 700; font-size: 14px; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .imggrid {{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 10px 0 8px;
      align-items:start;
    }}
    .imggrid.one {{ grid-template-columns: 1fr; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    details {{
      border-top: 1px dashed var(--border);
      margin-top: 10px;
      padding-top: 10px;
    }}
    summary {{ cursor:pointer; color: var(--accent); font-size: 13px; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #0d0d13;
      border: 1px solid var(--border);
      padding: 10px;
      border-radius: 10px;
      color: #e9e9f2;
      font-size: 12px;
      line-height: 1.35;
    }}
    .twocol {{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    @media (max-width: 860px) {{
      .twocol {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>CSWM-Bench Viewer (image track)</h1>
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
  <div class="grid" id="grid">
    {''.join(cards)}
  </div>
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
    print("Open it in a browser (local): file://" + str(out_path))


if __name__ == "__main__":
    main()

