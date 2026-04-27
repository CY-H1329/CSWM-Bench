#!/usr/bin/env python3
"""
Build an HTML viewer for I2V runs (SVD on CSWM).

Input: a run_dir containing manifest.jsonl
Output: <run_dir>/viewer/index.html
"""

from __future__ import annotations

import argparse
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


def _video_tag(rel_path: str) -> str:
    src = f"../../../{rel_path}"
    return (
        f'<video src="{html.escape(src)}" controls loop muted playsinline '
        f'style="width:100%;border-radius:10px;border:1px solid #2a2a2a;"></video>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., results/runs/i2v/svd/20260427_123456")
    args = ap.parse_args()

    run_dir = (ROOT / args.run_dir).resolve() if not str(args.run_dir).startswith("/") else Path(args.run_dir)
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}")

    rows = _read_jsonl(manifest)
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    cards = []
    for ex_id, jobs in sorted(by_id.items()):
        task = jobs[0].get("task", "")
        jobs = sorted(jobs, key=lambda x: x.get("job", ""))
        cols = []
        for j in jobs:
            cols.append(
                f"""
                <div>
                  <div class="label">{html.escape(j.get('job',''))}</div>
                  {_video_tag(j.get('video',''))}
                  <details><summary>prompt</summary><pre>{html.escape(j.get('prompt',''))}</pre></details>
                </div>
                """
            )
        grid_class = "one" if len(cols) == 1 else "two"
        cards.append(
            f"""
            <div class="card">
              <div class="title">{html.escape(ex_id)}</div>
              <div class="meta">task: <b>{html.escape(task)}</b></div>
              <div class="grid {grid_class}">{''.join(cols)}</div>
            </div>
            """
        )

    out_dir = run_dir / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    doc = f"""<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>I2V Viewer</title>
  <style>
    :root {{ --bg:#0b0b0f;--panel:#111117;--text:#e9e9f2;--muted:#a8a8b6;--border:#242430;--accent:#8ad1ff; }}
    body {{ background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui; margin:0; padding:24px; }}
    h1 {{ margin:0 0 10px; font-size:18px; }}
    .muted {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
    .wrap {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; }}
    .title {{ font-weight:700; font-size:14px; }}
    .meta {{ color:var(--muted); font-size:12px; margin-bottom:10px; }}
    .grid {{ display:grid; gap:10px; }}
    .grid.two {{ grid-template-columns:1fr 1fr; }}
    .grid.one {{ grid-template-columns:1fr; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    details {{ margin-top:8px; }}
    summary {{ cursor:pointer; color:var(--accent); font-size:13px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#0d0d13; border:1px solid var(--border); padding:10px; border-radius:10px; font-size:12px; }}
  </style>
</head>
<body>
  <h1>I2V Viewer (SVD on CSWM)</h1>
  <div class="muted">run: {html.escape(str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir))}</div>
  <div class="wrap">{''.join(cards)}</div>
</body></html>
"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote viewer: {out_path}")


if __name__ == "__main__":
    main()

