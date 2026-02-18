#!/usr/bin/env python3
"""
CV-Bench + 3DSRBench 결과를 표로 정리.

1. aggregate 실행 (cvbench, 3dsrbench)
2. results_summary/SUMMARY_TABLES.md 생성
3. aggregate CSV/JSON를 results_summary/로 복사

H100에서 실행 후 push → 로컬에서 pull

Usage:
  python scripts/compile_results_tables.py
  python scripts/compile_results_tables.py --results_dir /path/to/results
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
SUMMARY_ROOT = ROOT / "results_summary"


def run_aggregate_cvbench(results_dir: Path):
    """Run CV-Bench aggregate. Returns path to cvbench_by_category.csv or None."""
    script = ROOT / "scripts/evals/cvbench/aggregate_category_results.py"
    if not script.exists():
        print(f"[!] {script} not found")
        return None
    out_csv = results_dir / "runs" / "cvbench" / "cvbench_by_category.csv"
    out_json = results_dir / "runs" / "cvbench" / "cvbench_by_category.json"
    try:
        subprocess.run(
            [sys.executable, str(script), "--dir", str(results_dir / "runs" / "cvbench"), "--output", str(out_csv)],
            check=True,
            capture_output=True,
        )
        return out_csv if out_csv.exists() else out_json if out_json.exists() else None
    except subprocess.CalledProcessError as e:
        print(f"[!] CV-Bench aggregate failed: {e}")
        return None


def run_aggregate_3dsrbench(results_dir: Path):
    """Run 3DSRBench aggregate. Returns path to category_performance.csv or None."""
    script = ROOT / "scripts/evals/3dsrbench/aggregate_category_performance.py"
    if not script.exists():
        print(f"[!] {script} not found")
        return None
    api_base = results_dir / "runs" / "3dsrbench" / "api_models"
    if not api_base.exists():
        return None
    # Find latest or full_dataset
    subdirs = sorted(api_base.iterdir(), key=lambda p: p.name, reverse=True)
    for sub in subdirs:
        if sub.is_dir():
            out_csv = sub / "category_performance.csv"
            try:
                subprocess.run(
                    [sys.executable, str(script), "--dir", str(sub), "--mode", "api", "--output", str(out_csv)],
                    check=True,
                    capture_output=True,
                )
                return out_csv if out_csv.exists() else None
            except subprocess.CalledProcessError:
                continue
    return None


def load_cvbench_tables(results_dir: Path) -> dict:
    """Load CV-Bench results. Run aggregate if needed."""
    cv_csv = results_dir / "runs" / "cvbench" / "cvbench_by_category.csv"
    cv_json = results_dir / "runs" / "cvbench" / "cvbench_by_category.json"
    if not cv_csv.exists() and not cv_json.exists():
        run_aggregate_cvbench(results_dir)
    data = {}
    if cv_json.exists():
        with open(cv_json, encoding="utf-8") as f:
            data = json.load(f)
    elif cv_csv.exists():
        import csv
        with open(cv_csv, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                key = f"{row['model']}_{row['prompt_variant']}"
                if key not in data:
                    data[key] = {"model": row["model"], "prompt_variant": row["prompt_variant"], "per_category": {}}
                data[key]["per_category"][row["category"]] = {
                    "accuracy": float(row["accuracy"]),
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                }
    return data


def load_3dsrbench_tables(results_dir: Path) -> dict:
    """Load 3DSRBench results from results_summary or results."""
    data = {}
    # From results_summary (already gathered)
    for base in [SUMMARY_ROOT, results_dir]:
        api_path = base / "runs" / "3dsrbench" / "api_models" if base == results_dir else base / "3dsrbench" / "api_models"
        if not api_path.exists():
            api_path = base / "3dsrbench" / "api_models"
        if not api_path.exists():
            continue
        for ts_dir in sorted(api_path.iterdir(), key=lambda p: p.name, reverse=True):
            if not ts_dir.is_dir():
                continue
            for f in ts_dir.glob("category_*.json"):
                with open(f, encoding="utf-8") as fp:
                    j = json.load(fp)
                if isinstance(j, dict):
                    for model, per_cat in j.items():
                        if model not in data:
                            data[model] = per_cat
                        else:
                            for cat, v in per_cat.items():
                                if isinstance(v, dict) and "accuracy" in v:
                                    data[model][cat] = v
            # category_performance.json (from aggregate)
            perf = ts_dir / "category_performance.json"
            if perf.exists():
                with open(perf, encoding="utf-8") as fp:
                    j = json.load(fp)
                for model, per_cat in j.items():
                    data[model] = per_cat
            break  # use first (latest) dir
        if data:
            break
    return data


def load_gpu_results(results_dir: Path):
    """Load GPU results.json for CV-Bench and 3DSRBench."""
    cv_gpu, dsr_gpu = {}, {}
    for bench, out in [("cvbench", cv_gpu), ("3dsrbench", dsr_gpu)]:
        base = results_dir / "runs" / bench
        if not base.exists():
            base = SUMMARY_ROOT / ("cvbench" if bench == "cvbench" else "3dsrbench") / "gpu"
        if not base.exists():
            continue
        for model_dir in base.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                rj = run_dir / "results.json"
                if rj.exists():
                    with open(rj, encoding="utf-8") as f:
                        d = json.load(f)
                    key = f"{model_dir.name}_{run_dir.name}"
                    out[key] = d
    return cv_gpu, dsr_gpu


def build_markdown_tables(cv_data: dict, dsr_data: dict, cv_gpu: dict, dsr_gpu: dict) -> str:
    """Build SUMMARY_TABLES.md content."""
    lines = [
        "# CV-Bench & 3DSRBench Results Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## CV-Bench",
        "",
    ]

    # CV-Bench overall
    if cv_gpu or cv_data:
        lines.append("### Overall Accuracy")
        lines.append("")
        lines.append("| Model | With Prompt | Without Prompt |")
        lines.append("|-------|-------------|----------------|")
        models = sorted(set(
            (k.replace("_with_prompt", "").replace("_without_prompt", "").strip("_") for k in (list(cv_data.keys()) + list(cv_gpu.keys())))
        ))
        for m in models:
            wp, wop = "-", "-"
            for k, v in cv_data.items():
                if not isinstance(v, dict) or v.get("model") != m:
                    continue
                acc = v.get("overall", {}).get("accuracy")
                if acc is not None:
                    if v.get("prompt_variant") == "with_prompt":
                        wp = f"{acc:.2%}"
                    else:
                        wop = f"{acc:.2%}"
            for k, v in cv_gpu.items():
                if m not in k:
                    continue
                acc = v.get("accuracy") if isinstance(v, dict) else None
                if acc is not None:
                    if "with_prompt" in k:
                        wp = f"{acc:.2%}"
                    else:
                        wop = f"{acc:.2%}"
            lines.append(f"| {m} | {wp} | {wop} |")
        lines.append("")

    # CV-Bench by category
    if cv_data:
        lines.append("### CV-Bench by Category (Count, Relation, Depth, Distance)")
        lines.append("")
        cats = ["Count", "Relation", "Depth", "Distance"]
        models = sorted(set(d.get("model", k.split("_")[0]) for k, d in cv_data.items() if isinstance(d, dict)))
        for cat in cats:
            lines.append(f"#### {cat}")
            lines.append("")
            lines.append("| Model | With Prompt | Without Prompt |")
            lines.append("|-------|-------------|----------------|")
            for m in models:
                wp, wop = "-", "-"
                for k, d in cv_data.items():
                    if not isinstance(d, dict) or d.get("model") != m:
                        continue
                    pc = d.get("per_category", {})
                    v = pc.get(cat, {})
                    if isinstance(v, dict):
                        acc = v.get("accuracy")
                        if acc is not None:
                            if d.get("prompt_variant") == "with_prompt":
                                wp = f"{acc:.2%}"
                            else:
                                wop = f"{acc:.2%}"
                lines.append(f"| {m} | {wp} | {wop} |")
            lines.append("")
        lines.append("---")
        lines.append("")

    # 3DSRBench
    lines.append("## 3DSRBench")
    lines.append("")
    if dsr_gpu or dsr_data:
        lines.append("### Overall Accuracy")
        lines.append("")
        lines.append("| Model | With Prompt | Without Prompt |")
        lines.append("|-------|-------------|----------------|")
        models = sorted(set(
            k.replace("_with_prompt", "").replace("_without_prompt", "").replace("_1", "").strip("_")
            for k in (list(dsr_data.keys()) + list(dsr_gpu.keys()))
        ))
        for m in models:
            wp, wop = "-", "-"
            for k, v in dsr_gpu.items():
                if m in k and isinstance(v, dict):
                    acc = v.get("accuracy")
                    if acc is not None:
                        if "with_prompt" in k:
                            wp = f"{acc:.2%}"
                        else:
                            wop = f"{acc:.2%}"
            for k, v in dsr_data.items():
                if m in k and isinstance(v, dict):
                    total = sum(x.get("total", 0) for x in v.values() if isinstance(x, dict))
                    correct = sum(x.get("correct", 0) for x in v.values() if isinstance(x, dict))
                    if total:
                        acc = correct / total
                        if "with_prompt" in k:
                            wp = f"{acc:.2%}"
                        else:
                            wop = f"{acc:.2%}"
            lines.append(f"| {m} | {wp} | {wop} |")
        lines.append("")

    # 3DSRBench by category (simplified - top categories)
    if dsr_data:
        lines.append("### 3DSRBench by Category (12 categories)")
        lines.append("")
        cats_3d = [
            "location_above", "height_higher", "location_closer_to_camera", "multi_object_closer_to",
            "orientation_on_the_left", "multi_object_facing", "multi_object_same_direction",
            "orientation_in_front_of", "multi_object_viewpoint_towards_object", "orientation_viewpoint",
            "location_next_to", "multi_object_parallel",
        ]
        models = sorted(set(k.replace("_with_prompt_1", "").replace("_without_prompt_1", "") for k in dsr_data.keys()))
        lines.append("| Category | " + " | ".join(m[:15] for m in models[:6]) + " |")
        lines.append("|" + "---|" * (min(6, len(models)) + 1))
        for cat in cats_3d[:6]:  # first 6 for brevity
            row = [cat[:25]]
            for m in models[:6]:
                acc = "-"
                for k, v in dsr_data.items():
                    if m in k and isinstance(v, dict) and cat in v:
                        x = v[cat]
                        if isinstance(x, dict):
                            acc = f"{x.get('accuracy', 0):.2%}"
                row.append(acc)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("*Full category table in category_performance.csv*")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(SUMMARY_ROOT))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_root = Path(args.output)
    summary_root.mkdir(parents=True, exist_ok=True)

    print("1. Running CV-Bench aggregate...")
    run_aggregate_cvbench(results_dir)

    print("2. Running 3DSRBench aggregate...")
    run_aggregate_3dsrbench(results_dir)

    print("3. Loading data...")
    cv_data = load_cvbench_tables(results_dir)
    dsr_data = load_3dsrbench_tables(results_dir)
    cv_gpu, dsr_gpu = load_gpu_results(results_dir)

    print("4. Building SUMMARY_TABLES.md...")
    md = build_markdown_tables(cv_data, dsr_data, cv_gpu, dsr_gpu)
    out_md = summary_root / "SUMMARY_TABLES.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"  -> {out_md}")

    print("5. Copying aggregate outputs to results_summary...")
    for src in [
        results_dir / "runs" / "cvbench" / "cvbench_by_category.csv",
        results_dir / "runs" / "cvbench" / "cvbench_by_category.json",
    ]:
        if src.exists():
            dst = summary_root / src.name
            shutil.copy2(src, dst)
            print(f"  -> {dst}")

    api_3d = results_dir / "runs" / "3dsrbench" / "api_models"
    if api_3d.exists():
        for sub in sorted(api_3d.iterdir(), key=lambda p: p.name, reverse=True):
            if sub.is_dir():
                for f in ["category_performance.csv", "category_performance.json"]:
                    src = sub / f
                    if src.exists():
                        dst = summary_root / "3dsrbench" / sub.name / f
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        print(f"  -> {dst}")
                break

    print("\nDone. Run on H100:")
    print("  git add results_summary/")
    print("  git commit -m 'Results: CV-Bench + 3DSRBench tables'")
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    exit(main())
