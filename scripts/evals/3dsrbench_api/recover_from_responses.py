#!/usr/bin/env python3
"""
Récupère details.jsonl et results.json à partir des fichiers responses/sample_*.txt.

Quand le script d'éval plante avant d'écrire details.jsonl (ex: dossier renommé),
les sample_*.txt ont déjà été écrits. Ce script les parse et régénère les sorties.

Usage:
  python scripts/evals/3dsrbench_api/recover_from_responses.py --dir results/runs/3dsrbench/api_models/20260217_060437/claude_sonnet_4_5_without_prompt

  # Ou si le dossier a été renommé :
  python scripts/evals/3dsrbench_api/recover_from_responses.py --dir /chemin/vers/claude_sonnet_4_5_without_prompt
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter
from typing import Optional

def _accuracy(pred_letters: list, gt_letters: list) -> float:
    if not pred_letters or len(pred_letters) != len(gt_letters):
        return 0.0
    return sum(p == g for p, g in zip(pred_letters, gt_letters)) / len(pred_letters)


def parse_sample_file(path: Path) -> Optional[dict]:
    """Parse sample_XXXXX.txt, return dict for details.jsonl."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    idx_match = re.search(r"sample_(\d+)\.txt", path.name)
    idx = int(idx_match.group(1)) if idx_match else 0

    query = ""
    gt = ""
    category_gt = ""
    pred_category = ""
    full_response = ""
    letter = ""

    parts = text.split("=== ")
    for part in parts:
        if part.startswith("QUERY ==="):
            query = part.replace("QUERY ===\n", "").split("\n\n=== ")[0].strip()
        elif part.startswith("GT ==="):
            gt = part.replace("GT ===\n", "").split("\n\n=== ")[0].strip()
        elif part.startswith("CATEGORY GT / PRED ==="):
            line = part.replace("CATEGORY GT / PRED ===\n", "").split("\n\n=== ")[0].strip()
            if " / " in line:
                category_gt, pred_category = line.split(" / ", 1)
                category_gt = category_gt.strip()
                pred_category = pred_category.strip()
        elif part.startswith("FULL RESPONSE ==="):
            full_response = part.replace("FULL RESPONSE ===\n", "").split("\n\n=== EXTRACTED PRED ===")[0].strip()
        elif part.startswith("EXTRACTED PRED ==="):
            letter = part.replace("EXTRACTED PRED ===\n", "").split("\n")[0].strip()

    return {
        "idx": idx,
        "query": query,
        "gt": gt,
        "pred": letter,
        "category_gt": category_gt,
        "pred_category": pred_category,
        "full_response": full_response,
    }


def main():
    parser = argparse.ArgumentParser(description="Récupère details.jsonl depuis responses/sample_*.txt")
    parser.add_argument("--dir", required=True, help="Dossier du run (contient responses/) ou chemin direct vers responses/")
    args = parser.parse_args()

    p = Path(args.dir).resolve()
    if (p / "responses").exists():
        base = p
        responses_dir = p / "responses"
    elif p.name == "responses" and p.exists():
        base = p.parent
        responses_dir = p
    else:
        print(f"[ERREUR] Indiquez le dossier du run (ex: .../claude_sonnet_4_5_without_prompt)")
        return 1
    if not responses_dir.exists():
        print(f"[ERREUR] responses/ introuvable dans {base}")
        return 1

    sample_files = sorted(responses_dir.glob("sample_*.txt"))
    if not sample_files:
        print(f"[ERREUR] Aucun sample_*.txt dans {responses_dir}")
        return 1

    print(f"Trouvé {len(sample_files)} samples")
    details = []
    for p in sample_files:
        d = parse_sample_file(p)
        if d:
            details.append(d)

    details.sort(key=lambda x: x["idx"])
    preds = [d.get("pred", "") for d in details]
    gt_list = [d.get("gt", "") for d in details]
    acc = _accuracy(preds, gt_list)

    cat_pairs = [(d.get("category_gt", ""), d.get("pred_category", "")) for d in details if d.get("category_gt")]
    cat_cls_acc = _accuracy([p[1] for p in cat_pairs], [p[0] for p in cat_pairs]) if cat_pairs else 0.0
    pred_dist = dict(sorted(Counter(p for p in preds if p).items()))

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    model_name = base.name
    with open(base / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "accuracy": acc,
            "n": len(details),
            "pred_distribution": pred_dist,
            "category_cls_accuracy": cat_cls_acc,
            "category_cls_n": len(cat_pairs),
        }, f, indent=2, ensure_ascii=False)

    print(f"  Accuracy: {acc:.4f} | Category Cls: {cat_cls_acc:.4f} | N={len(details)}")
    print(f"  Écrit: {base / 'details.jsonl'}")
    print(f"  Écrit: {base / 'results.json'}")
    return 0


if __name__ == "__main__":
    exit(main())
