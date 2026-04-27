#!/usr/bin/env python3
"""
HTML viewer for CSWM-PHYRE pairs.

Outputs:
  reports/cswmbench_phyre_viewer/index.html
"""

from __future__ import annotations

import html
import json
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


def _img_tag(rel_path: str, max_w: int = 520) -> str:
    src = f"../../{rel_path}"
    return f'<img src="{html.escape(src)}" style="max-width:{max_w}px;width:100%;border-radius:10px;border:1px solid #2a2a2a;" />'


def main() -> None:
    data_path = ROOT / "data" / "cswmbench_phyre" / "cswm_phyre_pairs.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}. Run build_cswm_phyre_pairs.py first.")

    items = _read_jsonl(data_path)
    out_dir = ROOT / "reports" / "cswmbench_phyre_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    cards = []
    for ex in items[:5000]:
        ex_id = ex.get("id", "")
        task_id = ex.get("task_id", "")
        img = ex.get("image", "")
        a1 = ex.get("a1", [])
        a2 = ex.get("a2", [])
        gt = ex.get("gt", {}) or {}
        gt_pre = html.escape(json.dumps(gt, ensure_ascii=False, indent=2))
        a_pre = html.escape(json.dumps({"a1": a1, "a2": a2}, ensure_ascii=False, indent=2))
        cards.append(
            f"""
            <div class="card">
              <div class="top">
                <div>
                  <div class="title">{html.escape(str(ex_id))}</div>
                  <div class="meta">task_id: <b>{html.escape(str(task_id))}</b></div>
                </div>
              </div>
              <div class="imggrid one">{_img_tag(img)}</div>
              <details>
                <summary>Actions + GT</summary>
                <div class="twocol">
                  <div><div class="label">actions</div><pre>{a_pre}</pre></div>
                  <div><div class="label">gt</div><pre>{gt_pre}</pre></div>
                </div>
              </details>
            </div>
            """
        )

    html_doc = f"""<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CSWM-PHYRE Viewer</title>
  <style>
    :root {{ --bg:#0b0b0f;--panel:#111117;--text:#e9e9f2;--muted:#a8a8b6;--border:#242430;--accent:#8ad1ff; }}
    body {{ background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui; margin:0; padding:24px; }}
    .header {{ display:flex;gap:16px;align-items:flex-end;justify-content:space-between;margin-bottom:18px; }}
    h1 {{ margin:0;font-size:18px; }}
    .muted {{ color:var(--muted);font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }}
    .card {{ background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px; }}
    .title {{ font-weight:700;font-size:14px; }}
    .meta {{ color:var(--muted);font-size:12px; }}
    .imggrid {{ display:grid; grid-template-columns:1fr; gap:10px; margin:10px 0 8px; }}
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
      <h1>CSWM-PHYRE Viewer</h1>
      <div class="muted">{len(items)} pairs · file: {html.escape(str(data_path.relative_to(ROOT)))}</div>
    </div>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body></html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote viewer: {out_path}")


if __name__ == "__main__":
    main()

