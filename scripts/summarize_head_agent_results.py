#!/usr/bin/env python3
"""
Head-Agent 선별 결과를 표로 정리.

results_summary/head_agent/ 또는 results/runs/head_agent/ 에서
CV-Bench, 3DSRBench category routing 결과를 읽어 표로 출력.

Usage:
  python scripts/summarize_head_agent_results.py
  python scripts/summarize_head_agent_results.py --dir results_summary
  python scripts/summarize_head_agent_results.py --dir results/runs --output results_summary/HEAD_AGENT_SUMMARY.md
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def find_head_agent_results(base: Path) -> dict:
    """
    Returns: {benchmark: {model: {accuracy, n, by_category}}}
    Supports: results_summary/head_agent/ or results/runs/head_agent/
    """
    base = Path(base).resolve()
    for candidate in [base / "head_agent", base / "runs" / "head_agent"]:
        if candidate.exists():
            head_base = candidate
            break
    else:
        return {}

    results = {}
    for benchmark in ["cvbench", "3dsrbench"]:
        cat_dir = head_base / benchmark / "category_routing"
        if not cat_dir.exists():
            continue
        # Latest timestamp
        ts_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
        if not ts_dirs:
            continue
        ts_dir = ts_dirs[0]
        results[benchmark] = {}
        for model_dir in ts_dir.iterdir():
            if not model_dir.is_dir():
                continue
            res_path = model_dir / "results.json"
            if not res_path.exists():
                continue
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results[benchmark][model_dir.name] = {
                    "accuracy": data.get("accuracy", 0),
                    "n": data.get("n", 0),
                    "by_category": data.get("by_category", {}),
                }
            except Exception:
                pass
    return results


def format_table(data: dict) -> str:
    """Format as markdown table."""
    lines = []
    lines.append("# Head-Agent 선별 결과 — Category Routing")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## 요약")
    lines.append("")

    if not data:
        lines.append("*No Head-Agent results found. Run on H100:*")
        lines.append("```bash")
        lines.append("python scripts/evals/head_agent_cvbench/run_eval_category_routing.py --max_samples 100")
        lines.append("python scripts/gather_results_summary.py")
        lines.append("git add results_summary/ && git commit -m 'Head-Agent results' && git push origin main")
        lines.append("```")
        return "\n".join(lines)

    models = []
    for b in data.values():
        models.extend(b.keys())
    models = sorted(set(models))

    # Main table: Model | CV-Bench | 3DSRBench | Average
    lines.append("| Model | CV-Bench | 3DSRBench | Average |")
    lines.append("|-------|----------|-----------|---------|")
    for model in models:
        cv_acc = data.get("cvbench", {}).get(model, {}).get("accuracy")
        dsr_acc = data.get("3dsrbench", {}).get(model, {}).get("accuracy")
        cv_str = f"{cv_acc:.2%}" if cv_acc is not None else "-"
        dsr_str = f"{dsr_acc:.2%}" if dsr_acc is not None else "-"
        vals = [v for v in [cv_acc, dsr_acc] if v is not None]
        avg = sum(vals) / len(vals) if vals else None
        avg_str = f"{avg:.2%}" if avg is not None else "-"
        lines.append(f"| {model} | {cv_str} | {dsr_str} | {avg_str} |")

    lines.append("")
    lines.append("## 상세 (per category)")
    lines.append("")

    for benchmark, models_data in data.items():
        lines.append(f"### {benchmark.upper()}")
        lines.append("")
        for model, mdata in sorted(models_data.items()):
            acc = mdata.get("accuracy", 0)
            n = mdata.get("n", 0)
            by_cat = mdata.get("by_category", {})
            lines.append(f"**{model}**: {acc:.2%} ({n} samples)")
            if by_cat:
                cat_str = ", ".join(f"{c}: {v:.2%}" for c, v in sorted(by_cat.items()))
                lines.append(f"  - {cat_str}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Head-Agent 결과 표 정리")
    parser.add_argument("--dir", default=str(ROOT / "results_summary"), help="results_summary 또는 results 디렉터리")
    parser.add_argument("--output", default=None, help="출력 파일 (미지정 시 stdout)")
    args = parser.parse_args()

    data = find_head_agent_results(Path(args.dir))
    table = format_table(data)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        print(f"Written: {out_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
