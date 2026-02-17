#!/usr/bin/env python3
"""
Gather key results from results/ into results_summary/ for paper submission.
Run on H100 after experiments. results_summary/ is tracked by git.

Usage:
  python scripts/gather_results_summary.py
  python scripts/gather_results_summary.py --results_dir /path/to/results
"""
import argparse
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
SUMMARY_ROOT = ROOT / "results_summary"

# Files to copy (relative to run dir)
API_FILES = ["category_claude.csv", "category_claude.json", "category_gpt4o.csv", "category_gpt4o.json", 
             "category_gemini.csv", "category_gemini.json", "summary.txt"]


def gather_results(results_dir: Path, summary_root: Path) -> int:
    """Copy key result files to results_summary/. Returns number of files copied."""
    results_dir = Path(results_dir).resolve()
    summary_root = Path(summary_root).resolve()
    if not results_dir.exists():
        print(f"[!] results_dir not found: {results_dir}")
        return 0

    n = 0
    # 3DSRBench API
    api_base = results_dir / "runs" / "3dsrbench" / "api_models"
    if api_base.exists():
        # Find latest timestamp or full_dataset
        for sub in sorted(api_base.iterdir(), key=lambda p: p.name, reverse=True):
            if not sub.is_dir():
                continue
            for fname in API_FILES:
                src = sub / fname
                if src.exists():
                    dst = summary_root / "3dsrbench" / "api_models" / sub.name / fname
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"  {src.relative_to(results_dir)} -> {dst.relative_to(ROOT)}")
                    n += 1

    # 3DSRBench GPU
    gpu_base = results_dir / "runs" / "3dsrbench"
    for model in ["qwen3_4b", "llava4d", "sa2va"]:
        model_dir = gpu_base / model
        if not model_dir.exists():
            continue
        for run_dir in model_dir.iterdir():
            if run_dir.is_dir():
                for fname in ["results.json"]:
                    src = run_dir / fname
                    if src.exists():
                        dst = summary_root / "3dsrbench" / "gpu" / model / run_dir.name / fname
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        print(f"  {src.relative_to(results_dir)} -> {dst.relative_to(ROOT)}")
                        n += 1

    return n


def main():
    parser = argparse.ArgumentParser(description="Gather results for paper submission")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS), help="Source results directory")
    parser.add_argument("--output", default=str(SUMMARY_ROOT), help="Output summary directory")
    args = parser.parse_args()

    summary_root = Path(args.output)
    summary_root.mkdir(parents=True, exist_ok=True)

    # Create README
    readme = summary_root / "README.md"
    readme.write_text(f"""# Results Summary

Generated: {datetime.now().isoformat()}

Aggregated results for paper submission. Raw data in `results/` on H100.

## Structure

- `3dsrbench/api_models/` — Claude, GPT-4o, Gemini (category CSV, summary)
- `3dsrbench/gpu/` — Qwen3, Sa2VA, LLaVA4D (results.json per run)
""", encoding="utf-8")
    print(f"  {readme.relative_to(ROOT)}")

    n = gather_results(Path(args.results_dir), summary_root)
    print(f"\nDone: {n} files copied to {summary_root}")
    return 0


if __name__ == "__main__":
    exit(main())
