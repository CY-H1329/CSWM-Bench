"""
MAS v2 Pipeline.

Train phase : iterate samples -> Head infers category -> select agents -> run specialists
              -> Final Reasoning Agent -> update score map using GT.
Test phase  : same flow but score map is frozen (no updates).

Each step: SharedMemory is cleared and rebuilt (cache semantics).

Models:
  HEAD            = Qwen3-VL-4B    (VLM,  image + text -> category)
  5 SPECIALISTS   = Qwen3/Sa2VA/LLaVA4D/SpatialRGPT/SpatialReasoner (VLM)
  FINAL REASONING = DeepSeek-R1    (text-only, SharedMemory + query -> answer)
"""
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from .config import SPECIALIST_LLMS, ROLES, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS
from .markdown_verbose import format_step_markdown, save_step_to_files
from .prompts import build_head_agent_prompt, build_role_prompt, build_final_reasoning_prompt
from .score_map import ScoreMap
from .score_map_updater import ScoreMapUpdater
from .shared_memory import SharedMemory


def _infer_answer_type(query: str) -> str:
    """Infer 'multiple_choice' or 'free_form' from query. Avoids circular import."""
    if not query or not query.strip():
        return "free_form"
    q = query.strip().upper()
    if "OPTIONS:" in q and ("(A)" in q or "(B)" in q):
        return "multiple_choice"
    return "free_form"

# C (Hybrid): tools only for explicit_3d and scene_graph
_ROLES_WITH_TOOLS = {"explicit_3d_representation", "scene_graph_construction"}

logger = logging.getLogger(__name__)


# ======================================================================
# Output parsers
# ======================================================================
def parse_category(raw: str, valid_categories: List[str]) -> str:
    """Best-effort match of Head Agent output to a valid category."""
    raw_clean = (raw or "").strip().lower()
    for cat in valid_categories:
        if cat.lower() == raw_clean:
            return cat
    for cat in valid_categories:
        if cat.lower() in raw_clean:
            return cat
    return valid_categories[0]


def parse_specialist_output(raw: str, answer_type: str = "multiple_choice") -> Tuple[str, str]:
    """Extract (answer, reason) from specialist output."""
    raw = (raw or "").strip()
    answer = ""
    reason = ""

    if answer_type == "free_form":
        # "Answer: 3", "Answer: two", "Answer: red"
        m = re.search(r"Answer\s*:\s*(.+?)(?=\n|Reason\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()[:100]
        if not answer:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            answer = lines[0][:100] if lines else ""
    else:
        # multiple_choice: (A)~(F)
        ans_m = re.search(r"Answer\s*:\s*\(?([A-F])\)?", raw, re.IGNORECASE)
        if ans_m:
            answer = f"({ans_m.group(1).upper()})"
        if not answer:
            fallback = re.search(r"\(([A-F])\)", raw)
            if fallback:
                answer = f"({fallback.group(1).upper()})"
            if not answer:
                for pat in [r"answer\s+is\s+\(?([A-F])\)?", r"choose\s+\(?([A-F])\)?",
                            r"therefore\s+\(?([A-F])\)?", r"option\s+\(?([A-F])\)?"]:
                    m = re.search(pat, raw, re.IGNORECASE)
                    if m:
                        answer = f"({m.group(1).upper()})"
                        break

    reason_m = re.search(r"Reason\s*:\s*(.+?)(?=\nAnswer\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if reason_m:
        reason = reason_m.group(1).strip()[:2000]

    return answer, reason


def parse_final_answer(raw: str, answer_type: str = "multiple_choice") -> str:
    """Extract the final answer from Reasoning Agent output."""
    raw = (raw or "").strip()
    if answer_type == "free_form":
        m = re.search(r"Answer\s*:\s*(.+?)(?=\n|Reason\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()[:100]
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[0][:100] if lines else ""
    # multiple_choice
    m = re.search(r"Answer\s*:\s*\(?([A-F])\)?", raw, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    m = re.search(r"\(([A-F])\)", raw)
    if m:
        return f"({m.group(1)})"
    return raw[:20]


# ======================================================================
# Single-step execution
# ======================================================================
def run_step(
    image: Image.Image,
    query: str,
    gt: Optional[str],
    step: int,
    total_steps: int,
    score_map: ScoreMap,
    head_generate: Callable[[Image.Image, str], str],
    specialist_generate: Callable[[str, Image.Image, str], str],
    reasoning_generate: Callable,
    updater: Optional[ScoreMapUpdater] = None,
    update_scores: bool = True,
    shared_object_extraction: bool = True,
    use_vlm_reasoning: bool = False,
    answer_type: Optional[str] = None,
    force_category: Optional[str] = None,
    verbose_markdown: bool = False,
    save_step_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Execute one step of the MAS v2 pipeline.

    Categories are always the fixed ALL_CATEGORIES (16 types).
    The Head Agent classifies any question into the best-fit category.

    answer_type: 'multiple_choice' | 'free_form'. If None, inferred from query (Options: present -> multiple_choice).
    force_category: If set (e.g. "counting" for Count-only benchmark), skip Head Agent and use this category.
    """
    t0 = time.time()
    if answer_type is None:
        answer_type = _infer_answer_type(query)

    # 1. Head Agent -> category inference (or use force_category)
    head_raw = None
    if force_category and force_category in ALL_CATEGORIES:
        category = force_category
    else:
        head_prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
        head_raw = head_generate(image, head_prompt)
        category = parse_category(head_raw, ALL_CATEGORIES)

    # 2. Agent selection from score map
    assignments = score_map.select_agents(category, step)

    # 2.5 Precompute object_names once when shared (saves ~1 VLM call)
    object_names_cache = None
    if shared_object_extraction:
        for role, llm_name in assignments:
            if role in _ROLES_WITH_TOOLS:
                from src2.tools.object_extraction import extract_objects_from_image
                object_names_cache = extract_objects_from_image(image, specialist_generate, llm_name)
                break

    # 3. Run 3 specialist agents -> SharedMemory (C: tools for explicit_3d, scene_graph)
    shared_memory = SharedMemory()
    agent_details = []
    tool_output_cache = {}  # role -> tool output (computed once per role type)
    for role, llm_name in assignments:
        if role in _ROLES_WITH_TOOLS and role not in tool_output_cache:
            try:
                if shared_object_extraction:
                    object_names = object_names_cache if object_names_cache else None
                else:
                    from src2.tools.object_extraction import extract_objects_from_image
                    object_names = extract_objects_from_image(image, specialist_generate, llm_name)
                if role == "explicit_3d_representation":
                    from src2.tools import get_3d_representation
                    tool_output_cache[role] = get_3d_representation(
                        image, object_names=object_names
                    )
                elif role == "scene_graph_construction":
                    from src2.tools import get_scene_graph
                    tool_output_cache[role] = get_scene_graph(
                        image, object_names=object_names
                    )
            except Exception as e:
                logger.warning("Tool for %s failed: %s", role, e)
                tool_output_cache[role] = ""
        tool_output = tool_output_cache.get(role, None)
        role_prompt = build_role_prompt(role, query, tool_output=tool_output, answer_type=answer_type)
        raw_output = specialist_generate(llm_name, image, role_prompt)
        answer, reason = parse_specialist_output(raw_output, answer_type=answer_type)
        shared_memory.add(role, llm_name, answer, reason)
        raw_store = raw_output if save_step_dir else raw_output[:3000]
        entry = {
            "role": role,
            "llm_name": llm_name,
            "answer": answer,
            "reason": reason,
            "raw_output": raw_store,
        }
        if save_step_dir:
            obj_names = object_names_cache if object_names_cache else None
            entry["tool_output"] = (tool_output or "").strip()
            entry["object_names"] = obj_names  # list of str or None
            entry["role_prompt"] = role_prompt
        agent_details.append(entry)

    # 4. Final Reasoning Agent (DeepSeek-R1 text-only, or Qwen3-VL-8B image+text)
    reasoning_prompt = build_final_reasoning_prompt(
        query, shared_memory.to_prompt_text(), with_image=use_vlm_reasoning, answer_type=answer_type
    )
    reasoning_raw = reasoning_generate(reasoning_prompt, image=image)
    final_answer = parse_final_answer(reasoning_raw, answer_type=answer_type)

    # 5. Score map update (train phase only)
    if update_scores and updater and gt:
        updater.update(
            score_map=score_map,
            category=category,
            assignments=assignments,
            agent_results=shared_memory.get_entries(),
            final_answer=final_answer,
            gt=gt,
            step=step,
            total_steps=total_steps,
        )

    elapsed = time.time() - t0
    correct = _is_correct(final_answer, gt, answer_type) if gt else None
    result = {
        "step": step,
        "category": category,
        "assignments": [(r, l) for r, l in assignments],
        "agent_details": agent_details,
        "final_answer": final_answer,
        "gt": gt,
        "correct": correct,
        "reasoning_raw": reasoning_raw[:3000],
        "elapsed_sec": round(elapsed, 2),
    }
    if verbose_markdown:
        result["verbose_markdown"] = format_step_markdown(
            step=step,
            query=query,
            category=category,
            head_raw=head_raw,
            assignments=assignments,
            agent_details=agent_details,
            final_answer=final_answer or "",
            gt=gt,
            correct=correct,
            score_map=score_map,
        )
    if save_step_dir:
        save_step_to_files(
            output_dir=Path(save_step_dir),
            step=step,
            query=query,
            category=category,
            head_raw=head_raw,
            assignments=assignments,
            agent_details=agent_details,
            final_answer=final_answer or "",
            gt=gt,
            correct=correct,
            score_map=score_map,
            reasoning_raw=reasoning_raw,
        )
    return result


# ======================================================================
# Train / Test phases
# ======================================================================
def run_train(
    dataset,
    benchmark: str,
    score_map: ScoreMap,
    head_generate: Callable,
    specialist_generate: Callable,
    reasoning_generate: Callable,
    updater: Optional[ScoreMapUpdater] = None,
    get_image_fn: Callable = None,
    get_prompt_fn: Callable = None,
    get_answer_fn: Callable = None,
    seed: int = 42,
    use_vlm_reasoning: bool = False,
) -> List[Dict]:
    """Train phase: iterate over dataset, update score map with GT.

    Categories are always the fixed ALL_CATEGORIES (16 types).
    """
    from src2.benchmarks.loaders import (
        get_benchmark_image, get_benchmark_prompt, get_benchmark_answer,
    )
    _img = get_image_fn or (lambda ex: get_benchmark_image(ex, benchmark))
    _prompt = get_prompt_fn or (lambda ex: get_benchmark_prompt(ex, benchmark))
    _answer = get_answer_fn or (lambda ex: get_benchmark_answer(ex, benchmark))

    updater = updater or ScoreMapUpdater()
    total = len(dataset)
    results = []

    for step in range(total):
        example = dataset[step]
        image = _img(example)
        query = _prompt(example)
        gt = _answer(example)

        if image is None:
            logger.warning("Step %d: image is None, skipping.", step)
            continue

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=step,
            total_steps=total,
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=updater,
            update_scores=True,
            use_vlm_reasoning=use_vlm_reasoning,
        )
        results.append(result)

        if (step + 1) % 50 == 0 or step == total - 1:
            correct_so_far = sum(1 for r in results if r.get("correct"))
            logger.info(
                "Train step %d/%d | running acc: %.2f%%",
                step + 1, total,
                100.0 * correct_so_far / len(results),
            )

    return results


def run_test(
    dataset,
    benchmark: str,
    score_map: ScoreMap,
    head_generate: Callable,
    specialist_generate: Callable,
    reasoning_generate: Callable,
    get_image_fn: Callable = None,
    get_prompt_fn: Callable = None,
    get_answer_fn: Callable = None,
    random_agents: bool = False,
    use_vlm_reasoning: bool = False,
    verbose: bool = False,
    verbose_markdown: bool = False,
    verbose_minimal: bool = False,
    updater: "ScoreMapUpdater" = None,
    update_scores: bool = False,
    save_step_dir: Optional[Union[str, Path]] = None,
    max_steps: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_callback: Optional[Callable] = None,
) -> List[Dict]:
    """Test phase: iterate over dataset.

    Categories are always the fixed ALL_CATEGORIES (16 types).
    random_agents: If True, use step=0 for each sample (random agent selection).
    verbose: If True, log step/acc/cat/assign and scores (every 5 steps).
    updater + update_scores: If set, update score map after each sample (TTO).
    """
    from src2.benchmarks.loaders import (
        get_benchmark_image, get_benchmark_prompt, get_benchmark_answer,
        get_benchmark_category,
    )
    _img = get_image_fn or (lambda ex: get_benchmark_image(ex, benchmark))
    _prompt = get_prompt_fn or (lambda ex: get_benchmark_prompt(ex, benchmark))
    _answer = get_answer_fn or (lambda ex: get_benchmark_answer(ex, benchmark))

    total = len(dataset)
    if max_steps is not None:
        total = min(total, max_steps)
    results = []

    for step in range(total):
        example = dataset[step]
        image = _img(example)
        query = _prompt(example)
        gt = _answer(example)
        gt_category = get_benchmark_category(example, benchmark)

        if image is None:
            logger.warning("Test step %d: image is None, skipping.", step)
            continue

        use_step = 0 if random_agents else step + 1
        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=use_step,
            total_steps=total,
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=updater,
            update_scores=update_scores and updater is not None,
            use_vlm_reasoning=use_vlm_reasoning,
            verbose_markdown=verbose_markdown,
            save_step_dir=save_step_dir,
        )
        if gt_category is not None:
            result["gt_category"] = gt_category
        results.append(result)

        if verbose_markdown and result.get("verbose_markdown"):
            print(result["verbose_markdown"])

        correct_so_far = sum(1 for r in results if r.get("correct"))
        acc_pct = 100.0 * correct_so_far / len(results) if results else 0

        if verbose_minimal:
            ok = "O" if result.get("correct") else "X"
            print(f"  Step {step + 1}/{total} | {ok}")
        elif verbose:
            cat = result.get("category", "unknown")
            assign = result.get("assignments", [])
            logger.info(
                "  Step %d/%d | acc: %.1f%% | cat: %s | assign: %s",
                step + 1, total, acc_pct, cat, assign,
            )
            # scores: every 5 steps, or every step when total <= 10
            show_scores = (step + 1) % 5 == 0 or step == 0 or total <= 10
            if show_scores:
                maps = score_map.get_all_maps()
                logger.info("    scores (step %d): %s", step + 1, maps)
        elif (step + 1) % 50 == 0 or step == total - 1:
            logger.info(
                "Test step %d/%d | running acc: %.2f%%",
                step + 1, total, acc_pct,
            )

        if checkpoint_every and checkpoint_callback and (step + 1) % checkpoint_every == 0:
            checkpoint_callback(results, step + 1)

    return results


# ======================================================================
# Helpers
# ======================================================================
def _is_correct(pred: str, gt: str, answer_type: str) -> bool:
    """Compare predicted vs ground-truth answer."""
    if answer_type == "free_form":
        return _norm_free_form(pred) == _norm_free_form(gt)
    return _norm_letter(pred) == _norm_letter(gt)


def _norm_letter(s: str) -> str:
    """Normalize multiple-choice answer (A~F)."""
    s = (s or "").strip().upper()
    for c in "ABCDEF":
        if c in s or f"({c})" in s:
            return c
    return s


def _norm_free_form(s: str) -> str:
    """Normalize free-form answer for comparison (lowercase, strip, basic number handling)."""
    s = (str(s or "").strip().lower())
    if not s:
        return ""
    # Optional: "3" == "three" — keep simple for now
    return s


def compute_accuracy(results: List[Dict], use_gt_category: bool = True) -> Dict:
    """Compute overall and per-category accuracy from result dicts.

    use_gt_category: If True and results have gt_category (from dataset), use it for
        per_category breakdown (e.g. CV-Bench: Count, Relation, Depth, Distance).
        Otherwise use Head's predicted category (unified: spatial_relation, etc.).
    """
    total = 0
    correct = 0
    by_cat: Dict[str, Dict[str, int]] = {}

    for r in results:
        if r.get("correct") is None:
            continue
        total += 1
        if r["correct"]:
            correct += 1
        cat = (r.get("gt_category") if use_gt_category and r.get("gt_category") else None) or r.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1

    overall = correct / total if total > 0 else 0.0
    per_category = {
        cat: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for cat, v in sorted(by_cat.items())
    }
    return {
        "accuracy": overall,
        "correct": correct,
        "total": total,
        "per_category": per_category,
        "per_category_counts": by_cat,
    }


def _infer_answer_type_from_gt(gt: str) -> str:
    """Infer multiple_choice vs free_form from ground truth."""
    if not gt or not str(gt).strip():
        return "free_form"
    g = str(gt).strip().upper()
    if any(c in g for c in "ABCDEF"):
        return "multiple_choice"
    return "free_form"


def _normalize_cat_for_match(gt_cat: str, pred_cat: str) -> Tuple[str, str]:
    """Normalize categories for head-agent match (FINE_TO_UNIFIED mapping)."""
    from .config import FINE_TO_UNIFIED
    gt_n = (FINE_TO_UNIFIED.get(gt_cat, gt_cat) if gt_cat else "") or gt_cat
    pred_n = (FINE_TO_UNIFIED.get(pred_cat, pred_cat) if pred_cat else "") or pred_cat
    return (gt_n or "unknown", pred_n or "unknown")


def compute_per_module_accuracy(
    results: List[Dict],
    use_gt_category: bool = True,
) -> Dict:
    """Compute per-module (head-agent, specialists, reasoning-agent) accuracy.

    Returns:
        - head_agent: {overall_acc, correct, total, per_category}
        - specialists: {agent_name: {overall_acc, correct, total, per_category}}
        - reasoning_agent: {overall_acc, correct, total, per_category}
        - per_task_per_module: {category: {head_agent, specialist_X, reasoning_agent}}
    """
    from collections import defaultdict

    total = 0
    head_correct = 0
    head_by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    spec_by_agent: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    reason_correct = 0
    reason_by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        if r.get("correct") is None:
            continue
        total += 1
        gt = r.get("gt") or ""
        gt_cat = r.get("gt_category") if use_gt_category else None
        pred_cat = r.get("category", "unknown")
        cat = gt_cat or pred_cat

        answer_type = _infer_answer_type_from_gt(gt)

        # Head agent: category match
        if gt_cat:
            gt_n, pred_n = _normalize_cat_for_match(gt_cat, pred_cat)
            h_ok = gt_n == pred_n
            if h_ok:
                head_correct += 1
            head_by_cat[cat]["total"] += 1
            if h_ok:
                head_by_cat[cat]["correct"] += 1

        # Specialists: each agent's answer vs gt
        for ad in r.get("agent_details", []):
            agent = ad.get("llm_name", "unknown")
            ans = ad.get("answer", "")
            s_ok = _is_correct(ans, gt, answer_type)
            spec_by_agent[agent][cat]["total"] += 1
            if s_ok:
                spec_by_agent[agent][cat]["correct"] += 1

        # Reasoning agent: final answer vs gt
        final = r.get("final_answer", "")
        r_ok = r.get("correct", False)
        if r_ok:
            reason_correct += 1
        reason_by_cat[cat]["total"] += 1
        if r_ok:
            reason_by_cat[cat]["correct"] += 1

    def _acc(d: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        return {c: (v["correct"] / v["total"] if v["total"] > 0 else 0.0) for c, v in sorted(d.items())}

    head_per_cat = _acc(head_by_cat) if head_by_cat else {}
    head_overall = head_correct / total if total > 0 and head_correct is not None else 0.0
    head_total = sum(v["total"] for v in head_by_cat.values()) if head_by_cat else total

    specialists = {}
    for agent, by_cat in spec_by_agent.items():
        c_total = sum(v["total"] for v in by_cat.values())
        c_correct = sum(v["correct"] for v in by_cat.values())
        specialists[agent] = {
            "overall_acc": c_correct / c_total if c_total > 0 else 0.0,
            "correct": c_correct,
            "total": c_total,
            "per_category": _acc(dict(by_cat)),
        }

    reason_per_cat = _acc(reason_by_cat)
    reason_overall = reason_correct / total if total > 0 else 0.0

    # per_task_per_module: category -> {head_agent, specialist_X, reasoning_agent}
    all_cats = sorted(set(list(head_by_cat.keys()) + list(reason_by_cat.keys())))
    per_task_per_module = {}
    for c in all_cats:
        per_task_per_module[c] = {
            "head_agent": head_by_cat.get(c, {}).get("correct", 0) / max(1, head_by_cat.get(c, {}).get("total", 0)),
            "reasoning_agent": reason_by_cat.get(c, {}).get("correct", 0) / max(1, reason_by_cat.get(c, {}).get("total", 0)),
        }
        for agent in specialists:
            sc = spec_by_agent.get(agent, {}).get(c, {})
            per_task_per_module[c][f"specialist_{agent}"] = sc.get("correct", 0) / max(1, sc.get("total", 0))

    return {
        "total": total,
        "head_agent": {
            "overall_acc": head_overall,
            "correct": head_correct,
            "total": head_total,
            "per_category": head_per_cat,
        },
        "specialists": specialists,
        "reasoning_agent": {
            "overall_acc": reason_overall,
            "correct": reason_correct,
            "total": total,
            "per_category": reason_per_cat,
        },
        "per_task_per_module": per_task_per_module,
    }


def save_per_module_report(metrics: Dict, output_path: Union[str, Path]) -> None:
    """Save per-module (head/specialists/reasoning) metrics to JSON and Markdown."""
    import json
    from datetime import datetime

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # JSON (full)
    json_path = out if str(out).endswith(".json") else out.with_suffix(".json")
    to_save = {k: v for k, v in metrics.items() if k != "per_task_per_module"}
    to_save["per_task_per_module"] = metrics.get("per_task_per_module", {})
    to_save["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path.write_text(json.dumps(to_save, indent=2, ensure_ascii=False))

    # Markdown (readable)
    md_path = json_path.with_suffix(".md")
    lines = [
        "# Per-Module Accuracy Report",
        "",
        f"**Total samples:** {metrics.get('total', 0)}",
        "",
        "## Head Agent (category inference)",
        "",
    ]
    ha = metrics.get("head_agent", {})
    lines.append(f"- Overall: {ha.get('correct', 0)}/{ha.get('total', 0)} = {100*ha.get('overall_acc', 0):.1f}%")
    for cat, acc in sorted(ha.get("per_category", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  - {cat}: {100*acc:.1f}%")
    lines.append("")
    lines.append("## Specialists (per agent)")
    lines.append("")
    for agent, data in sorted(metrics.get("specialists", {}).items()):
        lines.append(f"### {agent}")
        lines.append(f"- Overall: {data.get('correct', 0)}/{data.get('total', 0)} = {100*data.get('overall_acc', 0):.1f}%")
        for cat, acc in sorted(data.get("per_category", {}).items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {100*acc:.1f}%")
        lines.append("")
    lines.append("## Reasoning Agent (final synthesis)")
    lines.append("")
    ra = metrics.get("reasoning_agent", {})
    lines.append(f"- Overall: {ra.get('correct', 0)}/{ra.get('total', 0)} = {100*ra.get('overall_acc', 0):.1f}%")
    for cat, acc in sorted(ra.get("per_category", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  - {cat}: {100*acc:.1f}%")
    lines.append("")
    lines.append("## Per-Task × Per-Module")
    lines.append("")
    ptm = metrics.get("per_task_per_module", {})
    if ptm:
        cats = sorted(ptm.keys())
        first_val = next(iter(ptm.values()), {})
        modules = ["head_agent", "reasoning_agent"] + sorted(
            k for k in first_val.keys() if k.startswith("specialist_")
        )
        lines.append("| Category | " + " | ".join(m.replace("specialist_", "") for m in modules) + " |")
        lines.append("|" + "----------|" * len(modules) + "|")
        for cat in cats:
            row = [f"{100*ptm[cat].get(m, 0):.1f}%" for m in modules]
            lines.append(f"| {cat} | " + " | ".join(row) + " |")
    md_path.write_text("\n".join(lines))
