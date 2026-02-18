#!/usr/bin/env python3
"""
Export MAS score evolution to Excel/CSV.

Reads mas_results.jsonl + summary.json from a comparison run, outputs:
1. score_evolution.csv: agent × category × turn (initial 0.5, then updates)
2. agent_cot_per_sample.txt: CoT from each agent per sample

Usage:
  python scripts/evals/export_mas_scores_to_excel.py results/comparison_qwen3_mas/20260218_111701
  python scripts/evals/export_mas_scores_to_excel.py /path/to/run_dir
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.mas.config import CANDIDATE_AGENTS, TASK_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Export MAS scores and CoT to Excel/CSV")
    parser.add_argument("run_dir", type=str, help="Path to comparison run (e.g. results/comparison_qwen3_mas/20260218_111701)")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: same as run_dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else run_dir

    # 1. Load summary (score_history)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    score_history = summary.get("score_history", [])
    if not score_history:
        # Fallback: build from mas_results.jsonl or results.jsonl (score_table_after_turn)
        mas_path = run_dir / "mas_results.jsonl"
        if not mas_path.exists():
            mas_path = run_dir / "results.jsonl"
        if mas_path.exists():
            score_history = [{}]
            with open(mas_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    st = r.get("score_table_after_turn", {})
                    if st:
                        score_history.append(st)
            # First entry: initial (all 0.5)
            init = {m: {c: 0.5 for c in TASK_CATEGORIES} for m in CANDIDATE_AGENTS}
            if len(score_history) > 1:
                score_history[0] = init
            elif len(score_history) == 1 and not score_history[0]:
                score_history[0] = init
        else:
            print("WARNING: No score_history. Using initial 0.5 only.")
            init = {m: {c: 0.5 for c in TASK_CATEGORIES} for m in CANDIDATE_AGENTS}
            score_history = [init]

    # 2. Export score_evolution.csv
    # Format: turn, agent, depth, distance, relation, existence, count, instance_location, orientation, size, reach
    csv_path = out_dir / "score_evolution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["turn", "agent"] + TASK_CATEGORIES
        writer.writerow(header)
        for turn, st in enumerate(score_history):
            for agent in CANDIDATE_AGENTS:
                row = [turn, agent]
                for cat in TASK_CATEGORIES:
                    val = st.get(agent, {}).get(cat, 0.5)
                    row.append(f"{val:.2f}")
                writer.writerow(row)
    print(f"Saved: {csv_path}")

    # 3. Export agent × category pivot (one row per turn, wide format for Excel)
    # Rows: turn 0, turn 1, ... | Cols: agent_category (e.g. qwen3_4b_orientation)
    pivot_path = out_dir / "score_evolution_pivot.csv"
    with open(pivot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cols = [f"{a}_{c}" for a in CANDIDATE_AGENTS for c in TASK_CATEGORIES]
        writer.writerow(["turn"] + cols)
        for turn, st in enumerate(score_history):
            row = [turn]
            for agent in CANDIDATE_AGENTS:
                for cat in TASK_CATEGORIES:
                    val = st.get(agent, {}).get(cat, 0.5)
                    row.append(f"{val:.2f}")
            writer.writerow(row)
    print(f"Saved: {pivot_path}")

    # 4. Export CoT per sample
    cot_path = out_dir / "agent_cot_per_sample.txt"
    mas_path = run_dir / "mas_results.jsonl"
    if not mas_path.exists():
        mas_path = run_dir / "results.jsonl"
    if mas_path.exists():
        with open(mas_path, "r") as rf:
            lines = rf.readlines()
        with open(cot_path, "w", encoding="utf-8") as wf:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                idx = r.get("idx", "?")
                gt = r.get("gt", "?")
                pred = r.get("pred", "?")
                correct = r.get("correct", False)
                cat = r.get("predicted_category", "?")
                agents = r.get("selected_agents", [])
                agent_results = r.get("agent_results", [])
                reasoning = r.get("reasoning_justification", "")

                wf.write("=" * 80 + "\n")
                wf.write(f"Sample {idx} | GT={gt} | MAS pred={pred} | Correct={correct} | Category={cat}\n")
                wf.write(f"Selected agents: {agents}\n")
                wf.write("=" * 80 + "\n\n")

                for ar in agent_results:
                    name = ar.get("agent_name", "?")
                    strategy = ar.get("strategy", "")
                    cot = ar.get("cot", "")
                    answer = ar.get("answer", "")
                    confidence = ar.get("confidence", "")
                    log = ar.get("log", "")
                    wf.write(f"--- Agent: {name} ---\n")
                    wf.write(f"Strategy: {strategy}\n")
                    wf.write(f"CoT: {cot}\n")
                    wf.write(f"Answer: {answer} | Confidence: {confidence}\n")
                    wf.write(f"Log: {log}\n\n")

                wf.write(f"Reasoning justification: {reasoning}\n\n")
        print(f"Saved: {cot_path}")
    else:
        print(f"WARNING: {mas_path} not found, skipping CoT export")

    print("\nDone. Open score_evolution.csv or score_evolution_pivot.csv in Excel.")


if __name__ == "__main__":
    main()
