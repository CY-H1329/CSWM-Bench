#!/usr/bin/env python3
"""
STVQA-7K train/val split에서 카테고리별 샘플 개수 출력.
데이터셋이 어떻게 나뉘어 있는지 확인용.

Usage:
  python scripts/show_category_distribution.py
  python scripts/show_category_distribution.py --split val
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    ds = load_dataset("OX-PIXL/STVQA-7K", split=args.split)
    cats = [ds[i].get("category") or "unknown" for i in range(len(ds))]
    cnt = Counter(cats)

    print(f"=== STVQA-7K {args.split} split: 카테고리별 개수 ===")
    print(f"총 샘플: {len(ds)}")
    print(f"카테고리 수: {len(cnt)}")
    print()
    for c in sorted(cnt.keys()):
        pct = 100 * cnt[c] / len(ds)
        print(f"  {c}: {cnt[c]:4d}  ({pct:5.1f}%)")
    print()
    print(f"합계: {sum(cnt.values())}")


if __name__ == "__main__":
    main()
