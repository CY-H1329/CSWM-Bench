"""
Verbose markdown formatting for MAS v2 pipeline steps.

Provides structured, readable output for --verbose_markdown.
Also supports saving step data to text files (routing, per-agent CoT, final).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .score_map import ScoreMap

# Role display names for markdown / file naming
ROLE_DISPLAY = {
    "direct_visual_heuristic": "heuristic",
    "explicit_3d_representation": "3D representation",
    "scene_graph_construction": "2D scene graph",
}


def _role_display(role: str) -> str:
    return ROLE_DISPLAY.get(role, role)


def _role_filename(role: str) -> str:
    """Short name for filenames."""
    m = {
        "direct_visual_heuristic": "heuristic",
        "explicit_3d_representation": "3d",
        "scene_graph_construction": "scenegraph",
    }
    return m.get(role, role[:8])


def _block(text: str, indent: str = "> ") -> str:
    """Wrap text in blockquote for visual separation."""
    if not text.strip():
        return ""
    return indent + text.strip().replace("\n", "\n" + indent)


def format_step_markdown(
    step: int,
    query: str,
    category: str,
    head_raw: Optional[str],
    assignments: List[Tuple[str, str]],
    agent_details: List[Dict[str, Any]],
    final_answer: str,
    gt: Optional[str],
    correct: Optional[bool],
    score_map: ScoreMap,
) -> str:
    """Build a clean, structured markdown block for one pipeline step."""
    lines = []

    # --- Section 1: Question ---
    lines.append("")
    lines.append("## Step {} — Question".format(step + 1))
    lines.append("")
    lines.append("**Query**")
    lines.append("")
    lines.append(_block(query))
    lines.append("")

    # --- Section 2: Head Agent ---
    lines.append("### Head Agent → Category")
    lines.append("")
    if head_raw:
        lines.append("**Raw output**")
        lines.append("")
        lines.append(_block(head_raw))
        lines.append("")
    lines.append("**Grouping:** `{}`".format(category))
    lines.append("")

    # --- Section 3: Score list & selection ---
    lines.append("### Confidence Score & Selection")
    lines.append("")
    lines.append(score_map.to_markdown_list(category))
    lines.append("")
    lines.append("**Selected agents**")
    for role, llm_name in assignments:
        s = score_map.get_score(category, role, llm_name)
        lines.append("  - {} → **{}** `{:.3f}`".format(_role_display(role), llm_name, s))
    lines.append("")

    # --- Section 4: Agent outputs ---
    lines.append("### Agent Outputs")
    lines.append("")
    for ad in agent_details:
        ans = (ad.get("answer") or "").strip()
        lines.append("**{}** ({})".format(_role_display(ad["role"]), ad["llm_name"]))
        if ans:
            lines.append("")
            lines.append(_block(ans))
        lines.append("")
    lines.append("")

    # --- Section 5: Final answer ---
    lines.append("### Final Answer")
    lines.append("")
    correct_str = "✓ Correct" if correct else "✗ Wrong" if correct is False else "N/A"
    pred = (final_answer or "").strip() or "—"
    gt_val = (gt or "").strip() or "—"
    lines.append("> **Pred**  `{}`".format(pred))
    lines.append("> **GT**  `{}`".format(gt_val))
    lines.append("> **Result**  {}".format(correct_str))
    lines.append("")

    # --- Section 6: Updated score map ---
    lines.append("### Updated Score Map")
    lines.append("")
    lines.append(score_map.to_markdown_all_categories())
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def save_step_to_files(
    output_dir: Path,
    step: int,
    query: str,
    category: str,
    head_raw: Optional[str],
    assignments: List[Tuple[str, str]],
    agent_details: List[Dict[str, Any]],
    final_answer: str,
    gt: Optional[str],
    correct: Optional[bool],
    score_map: ScoreMap,
    reasoning_raw: str = "",
) -> List[Path]:
    """Save step data to text files.

    Creates:
      1. step_{N:03d}_routing.txt  - Query, analysis, routing, agent+role selection
      2. step_{N:03d}_score_table.txt - Pretty ASCII score table (all categories)
      3. step_{N:03d}_agent_{role}_{llm}.txt - Per-agent: tools, prompt, CoT, answer
      4. step_{N:03d}_final.txt - Final reasoning CoT + answer + GT

    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = step + 1
    prefix = "step_{:03d}".format(n)
    written = []

    # 1. Routing file: Query, Head Agent analysis, detailed score & selection, role assignment
    routing_lines = [
        "=" * 70,
        "Step {} — Routing".format(n),
        "=" * 70,
        "",
        "Query",
        "-" * 50,
        query,
        "",
        "",
        "Head Agent — Analysis & Category Decision",
        "-" * 50,
        "Raw output (Head Agent 분석):",
        "",
        (head_raw or "(force_category)").strip(),
        "",
        "결정된 카테고리 (Grouping): {}".format(category),
        "",
        "",
        "Score Map — 현재 카테고리({})별 Agent × Role 점수".format(category),
        "-" * 50,
        "",
        score_map.to_markdown_list(category),
        "",
        "",
        "Agent Selection — Role별 에이전트 선정 과정 (상세)",
        "-" * 50,
    ]

    # Per-role: show all scores, selection logic, chosen agent
    cat_map = score_map.get_category_map(category) or {}
    for role, chosen_llm in assignments:
        role_disp = _role_display(role)
        routing_lines.append("")
        routing_lines.append("[Role] {} (역할: {})".format(role_disp, role))
        role_scores = cat_map.get(role, {})
        # All agents' scores for this role, sorted by score desc
        sorted_llms = sorted(
            score_map.llms,
            key=lambda ll: role_scores.get(ll, score_map.initial_score),
            reverse=True,
        )
        for llm in sorted_llms:
            s = score_map.get_score(category, role, llm)
            marker = "  ← selected" if llm == chosen_llm else ""
            routing_lines.append("    - {}: {:.4f}{}".format(llm, s, marker))
        if step == 0:
            routing_lines.append("    (step=0: random selection)")
        else:
            routing_lines.append("    (step>0: argmax → {})".format(chosen_llm))
        routing_lines.append("")

    routing_lines.append("")
    routing_lines.append("Role Assignment — 최종 할당")
    routing_lines.append("-" * 50)
    for role, llm_name in assignments:
        s = score_map.get_score(category, role, llm_name)
        routing_lines.append("  - {} → {} (score: {:.4f})".format(
            _role_display(role), llm_name, s
        ))

    routing_path = output_dir / "{}_routing.txt".format(prefix)
    routing_path.write_text("\n".join(routing_lines), encoding="utf-8")
    written.append(routing_path)

    # 2. Score table (pretty ASCII, all categories)
    score_table_path = output_dir / "{}_score_table.txt".format(prefix)
    score_table_path.write_text(
        "Step {} — Score Map\n".format(n) + "=" * 70 + "\n\n" + score_map.to_pretty_table(),
        encoding="utf-8",
    )
    written.append(score_table_path)

    # 3. Per-agent files: Query, tools (object_names, tool_output), prompt, CoT, answer
    for ad in agent_details:
        role = ad["role"]
        llm = ad["llm_name"]
        raw = (ad.get("raw_output") or "").strip()
        ans = (ad.get("answer") or "").strip()
        obj_names = ad.get("object_names")
        tool_out = (ad.get("tool_output") or "").strip()
        role_prompt = (ad.get("role_prompt") or "").strip()
        role_fn = _role_filename(role)
        agent_path = output_dir / "{}_agent_{}_{}.txt".format(prefix, role_fn, llm)
        agent_lines = [
            "=" * 70,
            "Step {} — Agent: {} ({}), Role: {}".format(n, llm, _role_display(role), role),
            "=" * 70,
            "",
            "Query",
            "-" * 50,
            query,
            "",
        ]
        # Tool inputs (object extraction, 3D/scene_graph output)
        if obj_names is not None or tool_out:
            agent_lines.append("Tool Inputs (object extraction, 3D/scene_graph)")
            agent_lines.append("-" * 50)
            if obj_names is not None:
                obj_str = ", ".join(obj_names) if isinstance(obj_names, (list, tuple)) else str(obj_names)
                agent_lines.append("Object names (extracted): [{}]".format(obj_str or "(none)"))
                agent_lines.append("")
            if tool_out:
                agent_lines.append("Tool output (passed to prompt):")
                agent_lines.append(tool_out)
                agent_lines.append("")
            agent_lines.append("")
        # Role prompt (full prompt sent to agent)
        if role_prompt:
            agent_lines.append("Role Prompt (full prompt sent to agent)")
            agent_lines.append("-" * 50)
            agent_lines.append(role_prompt)
            agent_lines.append("")
            agent_lines.append("")
        agent_lines.extend([
            "CoT (Chain of Thought)",
            "-" * 50,
            raw or "(empty)",
            "",
            "Answer",
            "-" * 50,
            ans or "(empty)",
        ])
        agent_path.write_text("\n".join(agent_lines), encoding="utf-8")
        written.append(agent_path)

    # 3. Final file: Query, reasoning CoT, final answer, GT
    correct_str = "Correct" if correct else "Wrong" if correct is False else "N/A"
    final_path = output_dir / "{}_final.txt".format(prefix)
    final_lines = [
        "=" * 70,
        "Step {} — Final Answer".format(n),
        "=" * 70,
        "",
        "Query",
        "-" * 50,
        query,
        "",
        "Final Reasoning (CoT)",
        "-" * 50,
        (reasoning_raw or "").strip() or "(empty)",
        "",
        "Result",
        "-" * 50,
        "Pred:   {}".format((final_answer or "").strip() or "—"),
        "GT:     {}".format((gt or "").strip() or "—"),
        "Correct: {}".format(correct_str),
    ]
    final_path.write_text("\n".join(final_lines), encoding="utf-8")
    written.append(final_path)

    return written
