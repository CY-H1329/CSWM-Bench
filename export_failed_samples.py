#!/usr/bin/env python3
"""
틀린 샘플만 따로 저장: 이미지 + 질문/옵션/정답/예측/category 등 전체 정보.
category별 폴더로 정리해서 파악·추가 정리하기 쉽게 함.

Usage:
  python export_failed_samples.py --run_dir results/20250109_123456
  python export_failed_samples.py --run_dir results/20250109_123456 --models qwen gpt
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

from src.data import load_stvqa


def safe_dirname(s: str) -> str:
    """폴더명으로 쓸 수 있게 정리."""
    if not s:
        return "unknown"
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "unknown"


def load_preds(run_dir: Path, model_name: str) -> list:
    path = run_dir / f"{model_name}_preds.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_dataset_for_run(run_dir: Path, config: dict, expected_len: int = None):
    """run 시 사용한 것과 동일한 데이터셋 로드 (dataset_info.json 우선). expected_len은 preds 개수."""
    dataset_info_path = run_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            info = json.load(f)
        return load_stvqa(
            dataset_name=info.get("dataset_name", "OX-PIXL/STVQA-7K"),
            split=info.get("split", "val"),
            max_samples=info.get("max_samples"),
            max_per_category=info.get("max_per_category"),
        )
    ds_cfg = config.get("dataset", {})
    name = ds_cfg.get("name", "OX-PIXL/STVQA-7K")
    # Ancien run sans dataset_info.json: deviner split/max_per_category en faisant matcher la taille
    if expected_len is not None:
        for split, max_per_cat in [("train", 100), ("train", None), ("val", None), ("val", 100)]:
            try:
                ds = load_stvqa(dataset_name=name, split=split, max_samples=None, max_per_category=max_per_cat)
                if len(ds) == expected_len:
                    print(f"[export] Using dataset: split={split}, max_per_category={max_per_cat}, len={len(ds)}")
                    return ds
            except Exception:
                continue
        raise SystemExit(
            f"No dataset matching preds length {expected_len}. "
            "Re-run run_eval.py (recent code writes dataset_info.json), then run export again."
        )
    return load_stvqa(
        dataset_name=name,
        split=ds_cfg.get("split", "val"),
        max_samples=ds_cfg.get("max_samples"),
        max_per_category=ds_cfg.get("max_per_category"),
    )


def export_failed(run_dir: Path, model_name: str, dataset, out_base: Path) -> int:
    """한 모델에 대해 틀린 샘플만 이미지+메타 저장. 반환: 실패 개수."""
    records = load_preds(run_dir, model_name)
    failed = [r for r in records if not r.get("correct", True)]

    model_out = out_base / model_name
    by_cat_dir = model_out / "by_category"
    model_out.mkdir(parents=True, exist_ok=True)
    by_cat_dir.mkdir(parents=True, exist_ok=True)

    if not failed:
        (model_out / "README.md").write_text(
            f"# {model_name}: 0 failed samples\nTotal: {len(records)} (all correct).\n",
            encoding="utf-8",
        )
        return 0

    manifest_path = model_out / "failed_manifest.jsonl"
    manifest_lines = []
    category_counts = defaultdict(int)

    for r in failed:
        idx = r["idx"]
        cat = r.get("category") or "unknown"
        cat_safe = safe_dirname(cat)
        category_counts[cat] += 1

        row = dataset[idx]
        img = row.get("images") or row.get("image")
        cat_dir = by_cat_dir / cat_safe
        cat_dir.mkdir(parents=True, exist_ok=True)
        img_path_rel = f"by_category/{cat_safe}/img_{idx:05d}.png"
        img_path_abs = model_out / img_path_rel
        if img is not None:
            try:
                img.save(img_path_abs)
            except Exception:
                img_path_rel = None
        # 정답 텍스트: answer_only가 A면 options[0] 등
        options = row.get("options") or []
        gt_letter = row.get("answer_only") or r.get("gt")
        pred_letter = r.get("pred")
        gt_text = ""
        pred_text = ""
        if options:
            try:
                gt_text = options[ord(gt_letter) - ord("A")] if gt_letter in "ABCD" and len(options) > ord(gt_letter) - ord("A") else ""
            except (IndexError, TypeError):
                pass
            try:
                pred_text = options[ord(pred_letter) - ord("A")] if pred_letter in "ABCD" and len(options) > ord(pred_letter) - ord("A") else ""
            except (IndexError, TypeError):
                pass

        entry = {
            "idx": idx,
            "category": cat,
            "question_only": row.get("question_only"),
            "question_with_options": row.get("question_with_options"),
            "options": options,
            "answer_gt": gt_letter,
            "answer_pred": pred_letter,
            "answer_text_gt": gt_text or row.get("answer_text"),
            "answer_text_pred": pred_text,
            "level": row.get("level"),
            "rating": row.get("rating"),
            "image_path": img_path_rel,
            "image_id": row.get("image_id"),
        }
        manifest_lines.append(json.dumps(entry, ensure_ascii=False))

        # 카테고리 폴더 안에 질문/모델답/GT 정리 (데이터셋처럼 보기 쉽게)
        info_lines = [
            "=== 질문 ===",
            (row.get("question_only") or row.get("question_with_options") or "").strip() or "(없음)",
            "",
            "=== 옵션 ===",
        ]
        for i, o in enumerate(options or []):
            label = chr(65 + i)
            info_lines.append(f"  ({label}) {o}")
        info_lines.extend([
            "",
            "=== 모델 예측 ===",
            f"  {pred_letter}  ({pred_text})" if pred_text else f"  {pred_letter}",
            "",
            "=== 정답 (GT) ===",
            f"  {gt_letter}  ({gt_text})" if gt_text else f"  {gt_letter}",
        ])
        info_path = cat_dir / f"img_{idx:05d}.txt"
        info_path.write_text("\n".join(info_lines), encoding="utf-8")

    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))

    # category별 요약
    summary = {
        "model": model_name,
        "num_failed": len(failed),
        "num_total": len(records),
        "by_category": dict(category_counts),
    }
    with open(model_out / "failed_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # README
    readme = [
        f"# Failed samples: {model_name}",
        f"- Total failed: {len(failed)} / {len(records)}",
        "",
        "## by_category/",
        "각 폴더는 task(category)별로 정리됨. 각 틀린 샘플마다:",
        "- `img_<idx>.png` : 이미지",
        "- `img_<idx>.txt` : 질문, 옵션, 모델 예측, 정답(GT)",
        "",
        "## failed_manifest.jsonl",
        "한 줄에 한 샘플 (JSON). 필드: idx, category, question_only, options, answer_gt, answer_pred, answer_text_gt, answer_text_pred, level, rating, image_path, image_id",
    ]
    (model_out / "README.md").write_text("\n".join(readme))

    return len(failed)


def main():
    parser = argparse.ArgumentParser(description="틀린 샘플만 이미지+전체 정보로 저장 (category별 정리)")
    parser.add_argument("--run_dir", type=str, required=True, help="run 디렉터리 (e.g. results/20250109_123456)")
    parser.add_argument("--models", nargs="+", default=None, help="내보낼 모델 (기본: run_dir 내 모든 *_preds.jsonl)")
    parser.add_argument("--out_dir", type=str, default=None, help="저장 위치 (기본: run_dir/failed_samples)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise SystemExit("config_snapshot.yaml not found. Run run_eval.py first.")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.models is None:
        args.models = [p.stem.replace("_preds", "") for p in run_dir.glob("*_preds.jsonl")]
    if not args.models:
        raise SystemExit("No *_preds.jsonl in run_dir. Run eval with save_predictions: true first.")

    out_base = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "failed_samples")
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_base}")

    expected_len = len(load_preds(run_dir, args.models[0]))
    if expected_len == 0:
        raise SystemExit(f"No predictions in {args.models[0]}_preds.jsonl")
    print(f"Preds count: {expected_len}")

    dataset = load_dataset_for_run(run_dir, config, expected_len=expected_len)
    if len(dataset) != expected_len:
        raise SystemExit(f"Dataset length {len(dataset)} != preds count {expected_len}. Export aborted.")
    print(f"Dataset: len={len(dataset)}")

    for model_name in args.models:
        n = export_failed(run_dir, model_name, dataset, out_base)
        print(f"  {model_name}: {n} failed -> {out_base / model_name}")

    (out_base / "README.md").write_text(
        f"# Failed samples export\n\nRun: {run_dir}\n\nEach model has by_category/<category>/ with img_<idx>.png and img_<idx>.txt (question, pred, GT).\n",
        encoding="utf-8",
    )
    print(f"Done. Failed samples under: {out_base}")
