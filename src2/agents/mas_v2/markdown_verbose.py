"""
Verbose markdown formatting for MAS v2 pipeline steps.

Provides structured, readable output for --verbose_markdown.
Also supports saving step data to text files (routing, selection, CoT, table).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .score_map import ScoreMap

# Section width for alignment
_SEP = "=" * 72
_SUB = "-" * 72

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


def _build_routing_section(
    n: int,
    query: str,
    category: str,
    head_raw: Optional[str],
    assignments: List[Tuple[str, str]],
    score_map: ScoreMap,
    step: int,
) -> List[str]:
    """Build routing + selection section (reusable)."""
    cat_map = score_map.get_category_map(category) or {}
    lines = [
        "",
        _SEP,
        "  STEP {} — ROUTING & SELECTION".format(n),
        _SEP,
        "",
        "┌─ Query ─────────────────────────────────────────────────────────────────",
        "",
        query,
        "",
        "└──────────────────────────────────────────────────────────────────────────",
        "",
        "",
        "┌─ Head Agent: Analysis & Category Decision ───────────────────────────────",
        "",
        "Raw output:",
        "",
        (head_raw or "(force_category)").strip(),
        "",
        "Decided category (grouping):  {}".format(category),
        "",
        "└──────────────────────────────────────────────────────────────────────────",
        "",
        "",
        "┌─ Score Map (category: {}) — Agent × Role ───────────────────────────────".format(category),
        "",
        score_map.to_markdown_list(category),
        "",
        "└──────────────────────────────────────────────────────────────────────────",
        "",
        "",
        "┌─ Agent Selection — Per-Role Process (detailed) ──────────────────────────",
        "",
    ]
    for role, chosen_llm in assignments:
        role_disp = _role_display(role)
        lines.append("  [Role] {} ({})".format(role_disp, role))
        role_scores = cat_map.get(role, {})
        sorted_llms = sorted(
            score_map.llms,
            key=lambda ll: role_scores.get(ll, score_map.initial_score),
            reverse=True,
        )
        for llm in sorted_llms:
            s = score_map.get_score(category, role, llm)
            marker = "  ← SELECTED" if llm == chosen_llm else ""
            lines.append("      {}  {:>10.4f}{}".format(llm, s, marker))
        lines.append("      → selection: {} (step={})".format(
            "random" if step == 0 else "argmax",
            step,
        ))
        lines.append("")
    lines.append("  Final assignment:")
    for role, llm_name in assignments:
        s = score_map.get_score(category, role, llm_name)
        lines.append("    {}  →  {}  (score: {:.4f})".format(_role_display(role), llm_name, s))
    lines.append("")
    lines.append("└──────────────────────────────────────────────────────────────────────────")
    return lines


def _build_agent_cot_section(ad: Dict[str, Any], query: str, n: int) -> List[str]:
    """Build per-agent CoT section."""
    role = ad["role"]
    llm = ad["llm_name"]
    raw = (ad.get("raw_output") or "").strip()
    ans = (ad.get("answer") or "").strip()
    reason = (ad.get("reason") or "").strip()
    obj_names = ad.get("object_names")
    tool_out = (ad.get("tool_output") or "").strip()
    role_prompt = (ad.get("role_prompt") or "").strip()
    lines = [
        "",
        _SUB,
        "  Agent: {}  |  Role: {}  ({})".format(llm, _role_display(role), role),
        _SUB,
        "",
        "  Query:",
        "",
        "    " + query.replace("\n", "\n    "),
        "",
    ]
    if obj_names is not None or tool_out:
        lines.append("  Tool inputs:")
        if obj_names is not None:
            obj_str = ", ".join(obj_names) if isinstance(obj_names, (list, tuple)) else str(obj_names)
            lines.append("    Object names: [{}]".format(obj_str or "(none)"))
        if tool_out:
            lines.append("    Tool output:")
            for ln in tool_out.split("\n"):
                lines.append("      " + ln)
        lines.append("")
    if role_prompt:
        lines.append("  Role prompt (full):")
        lines.append("")
        for ln in role_prompt.split("\n"):
            lines.append("    " + ln)
        lines.append("")
    lines.append("  CoT (Chain of Thought) — full raw output:")
    lines.append("")
    for ln in (raw or "(empty)").split("\n"):
        lines.append("    " + ln)
    lines.append("")
    if reason:
        lines.append("  Extracted reason:")
        lines.append("")
        for ln in reason.split("\n"):
            lines.append("    " + ln)
        lines.append("")
    lines.append("  Answer:  {}".format(ans or "(empty)"))
    lines.append("")
    return lines


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
      1. step_{N:03d}_summary.txt  - Full: routing + selection + all CoTs + result + table
      2. step_{N:03d}_routing.txt  - Routing & selection only
      3. step_{N:03d}_score_table.txt - ASCII score table (all categories)
      4. step_{N:03d}_agent_{role}_{llm}.txt - Per-agent: tools, prompt, CoT, answer
      5. step_{N:03d}_final.txt - Final reasoning CoT + result

    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = step + 1
    prefix = "step_{:03d}".format(n)
    written = []
    correct_str = "Correct" if correct else "Wrong" if correct is False else "N/A"

    # --- 1. Summary: routing + selection + all agent CoTs + final + table ---
    summary_lines = [
        _SEP,
        "  STEP {} — COMPLETE LOG (routing + selection + CoT + result + table)".format(n),
        _SEP,
    ]
    summary_lines.extend(_build_routing_section(n, query, category, head_raw, assignments, score_map, step))
    summary_lines.extend([
        "",
        "",
        _SEP,
        "  STEP {} — AGENT OUTPUTS (CoT & Results)".format(n),
        _SEP,
    ])
    for ad in agent_details:
        summary_lines.extend(_build_agent_cot_section(ad, query, n))
    summary_lines.extend([
        "",
        "",
        _SEP,
        "  STEP {} — FINAL ANSWER".format(n),
        _SEP,
        "",
        "  Final reasoning (CoT):",
        "",
    ])
    for ln in (reasoning_raw or "").strip().split("\n") or ["(empty)"]:
        summary_lines.append("    " + ln)
    summary_lines.extend([
        "",
        "  Result:",
        "    Pred:    {}".format((final_answer or "").strip() or "—"),
        "    GT:      {}".format((gt or "").strip() or "—"),
        "    Correct: {}".format(correct_str),
        "",
        "",
        _SEP,
        "  STEP {} — UPDATED SCORE TABLE (all categories)".format(n),
        _SEP,
        "",
        score_map.to_pretty_table(),
        "",
        "",
        _SEP,
    ])
    summary_path = output_dir / "{}_summary.txt".format(prefix)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    written.append(summary_path)

    # --- 2. Routing only ---
    routing_lines = _build_routing_section(n, query, category, head_raw, assignments, score_map, step)
    routing_path = output_dir / "{}_routing.txt".format(prefix)
    routing_path.write_text("\n".join(routing_lines), encoding="utf-8")
    written.append(routing_path)

    # --- 3. Score table ---
    score_table_path = output_dir / "{}_score_table.txt".format(prefix)
    score_table_path.write_text(
        _SEP + "\n  STEP {} — SCORE TABLE (all categories)\n".format(n) + _SEP + "\n\n" + score_map.to_pretty_table(),
        encoding="utf-8",
    )
    written.append(score_table_path)

    # --- 4. Per-agent files ---
    for ad in agent_details:
        role = ad["role"]
        llm = ad["llm_name"]
        role_fn = _role_filename(role)
        agent_path = output_dir / "{}_agent_{}_{}.txt".format(prefix, role_fn, llm)
        agent_lines = [
            _SEP,
            "  STEP {} — Agent: {}  |  Role: {}  ({})".format(n, llm, _role_display(role), role),
            _SEP,
        ]
        agent_lines.extend(_build_agent_cot_section(ad, query, n))
        agent_path.write_text("\n".join(agent_lines), encoding="utf-8")
        written.append(agent_path)

    # --- 5. Final ---
    final_path = output_dir / "{}_final.txt".format(prefix)
    final_lines = [
        _SEP,
        "  STEP {} — FINAL ANSWER".format(n),
        _SEP,
        "",
        "  Query:",
        "",
        "    " + query.replace("\n", "\n    "),
        "",
        "",
        "  Final reasoning (CoT):",
        "",
    ]
    for ln in (reasoning_raw or "").strip().split("\n") or ["(empty)"]:
        final_lines.append("    " + ln)
    final_lines.extend([
        "",
        "",
        "  Result:",
        "    Pred:    {}".format((final_answer or "").strip() or "—"),
        "    GT:      {}".format((gt or "").strip() or "—"),
        "    Correct: {}".format(correct_str),
        "",
        _SEP,
    ])
    final_path.write_text("\n".join(final_lines), encoding="utf-8")
    written.append(final_path)

    return written
