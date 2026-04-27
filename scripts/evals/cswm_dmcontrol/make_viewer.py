#!/usr/bin/env python3
"""
Create a portable HTML viewer for the latest CSWM-DMControl run.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="results/runs/cswm_dmcontrol/<timestamp>")
    args = ap.parse_args()

    base = ROOT / "results" / "runs" / "cswm_dmcontrol"
    if args.run_dir:
        run_dir = (ROOT / args.run_dir).resolve()
    else:
        # pick newest
        runs = sorted([p for p in base.glob("*") if p.is_dir()])
        if not runs:
            raise SystemExit("No runs found. Run render_pairs.py first.")
        run_dir = runs[-1]

    details_path = run_dir / "details.jsonl"
    if not details_path.exists():
        raise SystemExit(f"Missing {details_path}")
    rows = _read_jsonl(details_path)

    out_dir = run_dir / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    cards = []
    for r in rows:
        cards.append(
            f"""
            <div class="card">
              <div class="title">{html.escape(r.get('id',''))}</div>
              <div class="meta">state_l2_final: <b>{r.get('state_l2_final',0):.4f}</b> · k={r.get('k')} · delta={r.get('delta')}</div>
              <div class="grid">
                <div>
                  <div class="label">A</div>
                  <video src="../../../{html.escape(r.get('video_A',''))}" controls loop muted playsinline></video>
                </div>
                <div>
                  <div class="label">A'</div>
                  <video src="../../../{html.escape(r.get('video_Aprime',''))}" controls loop muted playsinline></video>
                </div>
              </div>
            </div>
            """
        )

    doc = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CSWM-DMControl Viewer</title>
<style>
:root{{--bg:#0b0b0f;--panel:#111117;--text:#e9e9f2;--muted:#a8a8b6;--border:#242430;}}
body{{background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui;margin:0;padding:24px;}}
h1{{margin:0 0 10px;font-size:18px;}}
.muted{{color:var(--muted);font-size:13px;margin-bottom:16px;}}
.wrap{{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:14px;}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px;}}
.title{{font-weight:700;font-size:14px;}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:10px;}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
.label{{color:var(--muted);font-size:12px;margin-bottom:6px;}}
video{{width:100%;border-radius:10px;border:1px solid #2a2a2a;}}
</style></head>
<body>
<h1>CSWM-DMControl Viewer</h1>
<div class="muted">run: {html.escape(str(run_dir.relative_to(ROOT)))}</div>
<div class="wrap">{''.join(cards)}</div>
</body></html>
"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote viewer: {out_path}")


if __name__ == "__main__":
    main()

