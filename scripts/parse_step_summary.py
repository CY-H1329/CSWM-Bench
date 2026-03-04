#!/usr/bin/env python3
"""
Parse step summary (e.g. step_050_summary.txt) to extract category→(role→agent) assignments.
Uses select_agents_by_score logic: argmax per role, no agent reuse.
Output: JSON with fixed assignments for inference.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Role name mapping: file display → config
ROLE_FILE_TO_CONFIG = {
    "heuristic": "direct_visual_heuristic",
    "3D representation": "explicit_3d_representation",
    "2D scene graph": "scene_graph_construction",
}


def _select_agents_by_score(
    scores: Dict[str, Dict[str, float]],
    roles: List[str],
    agents: List[str],
) -> Dict[str, str]:
    """Per-role argmax, no agent reuse. Returns {role: agent}."""
    assignment: Dict[str, str] = {}
    used: set = set()
    for role in roles:
        best_agent, best_score = None, -1e9
        for agent in agents:
            if agent in used:
                continue
            s = scores.get(agent, {}).get(role, -1e9)
            if s > best_score:
                best_score = s
                best_agent = agent
        if best_agent is not None:
            assignment[role] = best_agent
            used.add(best_agent)
        elif agents:
            assignment[role] = agents[0]
    return assignment


def parse_step_summary(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse step summary file. Returns:
      {category: [(role_config, agent), ...]}
    """
    text = Path(path).read_text(encoding="utf-8")
    roles_config = ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]
    role_file_names = ["heuristic", "3D representation", "2D scene graph"]

    # Find "STEP N — UPDATED SCORE TABLE"
    table_start = text.find("UPDATED SCORE TABLE")
    if table_start < 0:
        raise ValueError("No 'UPDATED SCORE TABLE' section found")

    table_text = text[table_start:]
    assignments: Dict[str, List[Tuple[str, str]]] = {}

    # Split by "Category: X"
    cat_blocks = re.split(r"\nCategory:\s+(\w+)\s*\n", table_text)[1:]
    for i in range(0, len(cat_blocks), 2):
        cat_name = cat_blocks[i].strip()
        block = cat_blocks[i + 1] if i + 1 < len(cat_blocks) else ""

        # Parse agents and scores
        scores: Dict[str, Dict[str, float]] = {}
        for line in block.split("\n"):
            line = line.strip()
            if not line or "---" in line or line.strip().startswith("Agent"):
                continue
            parts = re.split(r"\s+\|\s+", line)
            if len(parts) < 4:
                continue
            agent = parts[0].strip()
            if not agent:
                continue
            role_scores = {}
            for j, role_file in enumerate(role_file_names):
                idx = j + 1
                if idx < len(parts):
                    val_str = parts[idx].strip()
                    try:
                        role_scores[role_file] = float(val_str)
                    except ValueError:
                        role_scores[role_file] = -1e9
            if agent:
                scores[agent] = role_scores

        if not scores:
            continue

        # select_agents_by_score
        role_to_agent = _select_agents_by_score(
            scores, role_file_names, list(scores.keys())
        )
        assignments[cat_name] = [
            (ROLE_FILE_TO_CONFIG.get(r, r), a) for r, a in role_to_agent.items()
        ]

    return assignments


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("summary_path", help="Path to step_XXX_summary.txt")
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    args = p.parse_args()

    assignments = parse_step_summary(args.summary_path)
    out = {
        "source": args.summary_path,
        "assignments": {cat: [(r, a) for r, a in lst] for cat, lst in assignments.items()},
    }
    json_str = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
