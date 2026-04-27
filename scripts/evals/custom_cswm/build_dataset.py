#!/usr/bin/env python3
"""
Build a 2-task custom CSWM dataset from user-provided images.

Outputs:
  data/custom_cswm/custom_cswm.jsonl
  data/custom_cswm/images/*.png
  reports/custom_cswm_viewer/index.html  (portable viewer)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "custom_cswm"
IMG_DIR = DATA_DIR / "images"


def _copy_image(src: Path, name: str) -> str:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMG_DIR / name
    shutil.copy2(src, dst)
    return dst.relative_to(ROOT).as_posix()


def _viewer(items: List[Dict[str, Any]]) -> None:
    # reuse the existing portable viewer builder logic (simple inline)
    import html

    out_dir = ROOT / "reports" / "custom_cswm_viewer"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    def img_tag(rel_path: str) -> str:
        # copy dataset images into viewer for portability
        src = (ROOT / rel_path).resolve()
        dst = images_dir / Path(rel_path).name
        if not dst.exists():
            shutil.copy2(src, dst)
        return f'<img src="images/{html.escape(dst.name)}" style="width:100%;max-width:520px;border-radius:10px;border:1px solid #2a2a2a;" />'

    cards = []
    for ex in items:
        imgs = ex.get("images", []) or []
        imgs_html = ""
        if len(imgs) == 2:
            imgs_html = (
                '<div class="imggrid">'
                f'<div><div class="label">case1</div>{img_tag(imgs[0])}</div>'
                f'<div><div class="label">case2</div>{img_tag(imgs[1])}</div>'
                "</div>"
            )
        elif len(imgs) == 1:
            imgs_html = f'<div class="imggrid one">{img_tag(imgs[0])}</div>'
        gt_pre = html.escape(json.dumps(ex.get("gt", {}), ensure_ascii=False, indent=2))
        prompt_pre = html.escape(ex.get("prompt", ""))
        cards.append(
            f"""
            <div class="card">
              <div class="title">{html.escape(ex.get('id',''))}</div>
              <div class="meta">task: <b>{html.escape(ex.get('task',''))}</b></div>
              {imgs_html}
              <details><summary>Prompt + GT</summary>
                <div class="twocol">
                  <div><div class="label">prompt</div><pre>{prompt_pre}</pre></div>
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
<title>Custom CSWM Viewer</title>
<style>
:root{{--bg:#0b0b0f;--panel:#111117;--text:#e9e9f2;--muted:#a8a8b6;--border:#242430;--accent:#8ad1ff;}}
body{{background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui;margin:0;padding:24px;}}
h1{{margin:0 0 12px;font-size:18px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px;}}
.title{{font-weight:700;font-size:14px;}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:10px;}}
.imggrid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0 8px;}}
.imggrid.one{{grid-template-columns:1fr;}}
.label{{color:var(--muted);font-size:12px;margin-bottom:6px;}}
summary{{cursor:pointer;color:var(--accent);font-size:13px;}}
pre{{white-space:pre-wrap;word-break:break-word;background:#0d0d13;border:1px solid var(--border);padding:10px;border-radius:10px;font-size:12px;}}
.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}}
@media(max-width:860px){{.twocol{{grid-template-columns:1fr;}}}}
</style></head>
<body>
<h1>Custom CSWM Viewer</h1>
<div class="grid">{''.join(cards)}</div>
</body></html>
"""
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")
    print(f"Wrote viewer: {out_dir / 'index.html'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--door_empty", required=True)
    ap.add_argument("--door_blocked", required=True)
    ap.add_argument("--cup", required=True)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    door_empty = Path(args.door_empty).expanduser().resolve()
    door_blocked = Path(args.door_blocked).expanduser().resolve()
    cup = Path(args.cup).expanduser().resolve()
    for p in (door_empty, door_blocked, cup):
        if not p.exists():
            raise SystemExit(f"Missing image: {p}")

    # Copy images into dataset folder
    p_door1 = _copy_image(door_empty, "door_case1.png")
    p_door2 = _copy_image(door_blocked, "door_case2.png")
    p_cup = _copy_image(cup, "cup.png")

    items: List[Dict[str, Any]] = []

    # Door task: 2 images, same action
    door_prompt = (
        "You are shown TWO images (case1, case2). Action (SAME for both cases): OPEN THE DOOR fully.\n"
        "Choose ONE option:\n"
        "(A) both clear (door opens freely in both cases)\n"
        "(B) both blocked (door cannot open in both cases)\n"
        "(C) case1 clear, case2 blocked\n"
        "(D) case1 blocked, case2 clear\n"
        "Return ONLY the letter: A/B/C/D."
    )
    items.append(
        {
            "id": "custom_door_001",
            "task": "door",
            "images": [p_door1, p_door2],
            "prompt": door_prompt,
            # GT left empty on purpose: you can fill after you inspect / decide your assumption.
            "gt": {"answer": ""},
        }
    )

    # Cup task: 1 image, 2 actions inside the question
    cup_prompt = (
        "You are shown ONE image. Consider two actions:\n"
        "- action1: move/push the cup 5cm to the right\n"
        "- action2: move/push the cup 30cm to the right\n\n"
        "Choose ONE option:\n"
        "(A) both safe (cup stays on the table for both actions)\n"
        "(B) both fall (cup falls off for both actions)\n"
        "(C) 5cm safe, 30cm fall\n"
        "Return ONLY the letter: A/B/C."
    )
    items.append(
        {
            "id": "custom_cup_001",
            "task": "cup",
            "images": [p_cup],
            "prompt": cup_prompt,
            "gt": {"answer": ""},
        }
    )

    out_jsonl = DATA_DIR / "custom_cswm.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote dataset: {out_jsonl}")

    _viewer(items)


if __name__ == "__main__":
    main()

