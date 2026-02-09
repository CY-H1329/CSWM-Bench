#!/usr/bin/env python3
"""
실패한 질문을 STVQA-7K task(category)별로 분석.
run_eval.py 실행 시 config에서 save_predictions: true 로 두고 돌린 뒤,
이 스크립트로 해당 run 디렉터리를 지정하면 됨.

Usage:
  python analyze_failures.py --run_dir results/20250109_123456
  python analyze_failures.py --run_dir results/20250109_123456 --models qwen gpt --output report.md
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


# STVQA-7K task 유형 (데이터셋 category와 동일)
TASK_NAMES = {
    "relation": "Spatial Relation (above, behind, near, on top of...)",
    "reach": "Reach / Physical interaction (holding, touching, carrying)",
    "size": "Comparative Size (larger, smaller, taller)",
    "orientation": "Orientation (direction from viewpoint)",
    "instance_location": "Instance Location (position in image frame)",
    "depth": "Depth (closer/farther from camera)",
    "distance": "Distance (to reference object)",
    "count": "Count (object counting)",
    "existence": "Existence (yes/no presence)",
}


def load_preds(run_dir: Path, model_name: str) -> list:
    path = run_dir / f"{model_name}_preds.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze_failures_by_task(records: list) -> dict:
    """실패한 샘플만 골라 category별로 묶고, 통계 계산."""
    failed = [r for r in records if not r.get("correct", True)]
    by_category = defaultdict(lambda: {"failed": [], "total": 0, "correct": 0})
    for r in records:
        cat = r.get("category") or "unknown"
        by_category[cat]["total"] += 1
        if r.get("correct", True):
            by_category[cat]["correct"] += 1
        else:
            by_category[cat]["failed"].append(r)
    return {
        "num_failed": len(failed),
        "num_total": len(records),
        "failure_rate": len(failed) / len(records) if records else 0,
        "by_category": dict(by_category),
    }


def format_report(model_name: str, stats: dict, verbose: bool = True) -> str:
    lines = [
        f"# Failure analysis: {model_name}",
        f"- Total: {stats['num_total']} | Failed: {stats['num_failed']} | Failure rate: {stats['failure_rate']:.1%}",
        "",
        "## STVQA-7K task (category) 설명",
    ]
    for k, v in TASK_NAMES.items():
        lines.append(f"- **{k}**: {v}")
    lines.extend([
        "",
        "## Per-task (category) breakdown",
        "",
        "| Task (category) | Total | Correct | Failed | Acc (%) |",
        "|------------------|------:|-------:|------:|-------:|",
    ]
    for cat in sorted(stats["by_category"].keys(), key=lambda c: (-stats["by_category"][c]["total"], c)):
        d = stats["by_category"][cat]
        total = d["total"]
        correct = d["correct"]
        failed = len(d["failed"])
        acc = (correct / total * 100) if total else 0
        lines.append(f"| {cat} | {total} | {correct} | {failed} | {acc:.1f} |")

    if verbose:
        lines.append("")
        lines.append("## Failed questions by task")
        for cat in sorted(stats["by_category"].keys(), key=lambda c: -len(stats["by_category"][c]["failed"])):
            d = stats["by_category"][cat]
            if not d["failed"]:
                continue
            lines.append("")
            lines.append(f"### {cat} ({len(d['failed'])} failed)")
            for r in d["failed"][:20]:  # 최대 20개만 표시
                q = (r.get("question_only") or "").strip() or "(no question text)"
                lines.append(f"- [idx={r.get('idx')}] {q[:120]}{'...' if len(q) > 120 else ''}")
                lines.append(f"  - GT: {r.get('gt')} | Pred: {r.get('pred')}")
            if len(d["failed"]) > 20:
                lines.append(f"- ... and {len(d['failed']) - 20} more.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="실패 질문을 task(category)별로 분석")
    parser.add_argument("--run_dir", type=str, required=True, help="run 디렉터리 (e.g. results/20250109_123456)")
    parser.add_argument("--models", nargs="+", default=None, help="분석할 모델들 (기본: 해당 run에 있는 모든 *_preds.jsonl)")
    parser.add_argument("--output", type=str, default=None, help="보고서 저장 경로 (디렉터리면 run_dir 안에 저장)")
    parser.add_argument("--no_verbose", action="store_true", help="실패 질문 목록 생략, 요약만")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    if args.models is None:
        args.models = [
            p.stem.replace("_preds", "")
            for p in run_dir.glob("*_preds.jsonl")
        ]
    if not args.models:
        raise SystemExit("No *_preds.jsonl found. Run eval with save_predictions: true first.")

    all_reports = []
    for model_name in args.models:
        records = load_preds(run_dir, model_name)
        if not records:
            print(f"[skip] {model_name}: no preds found")
            continue
        stats = analyze_failures_by_task(records)
        report = format_report(model_name, stats, verbose=not args.no_verbose)
        all_reports.append((model_name, report, stats))

        print(report)
        print()

        # JSON 요약 저장 (category별 정확도 등)
        summary_path = run_dir / f"failure_analysis_{model_name}.json"
        json_stats = {
            "model": model_name,
            "num_total": stats["num_total"],
            "num_failed": stats["num_failed"],
            "failure_rate": stats["failure_rate"],
            "by_category": {
                cat: {
                    "total": d["total"],
                    "correct": d["correct"],
                    "failed": len(d["failed"]),
                    "accuracy_pct": (d["correct"] / d["total"] * 100) if d["total"] else 0,
                }
                for cat, d in stats["by_category"].items()
            },
        }
        with open(summary_path, "w") as f:
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        print(f"Saved {summary_path}")

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() in (".md", ".txt"):
            with open(out_path, "w") as f:
                f.write("\n\n---\n\n".join(r[1] for r in all_reports))
            print(f"Report written to {out_path}")
        elif out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            for model_name, report, _ in all_reports:
                (out_path / f"failure_report_{model_name}.md").write_text(report)
            print(f"Reports written to {out_path}")


if __name__ == "__main__":
    main()
