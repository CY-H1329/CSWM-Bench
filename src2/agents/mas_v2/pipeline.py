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


def _save_timing_to_file(
    output_dir: Path,
    step: int,
    category: str,
    timing: Dict,
    assignments: List[Tuple[str, str]],
) -> None:
    """Save per-step timing and append to cumulative timing_summary.csv."""
    n = step + 1
    prefix = "step_{:03d}".format(n)

    # 1. Per-step timing file (step_XXX_timing.txt)
    lines = [
        "=" * 60,
        f"  STEP {n} — RUNTIME (seconds) by module",
        "=" * 60,
        "",
        f"  head_agent:        {timing.get('head_agent_sec', 0):.3f} s",
        f"  object_extraction: {timing.get('object_extraction_sec', 0):.3f} s",
        "",
        "  specialists:",
    ]
    for s in timing.get("specialists_sec", []):
        role_short = s.get("role", "").replace("direct_visual_heuristic", "heuristic").replace("explicit_3d_representation", "3d_repr").replace("scene_graph_construction", "scene_graph")
        lines.append(f"    - {role_short} ({s.get('llm', '')}): {s.get('sec', 0):.3f} s")
    lines.extend([
        "",
        f"  reasoning_agent:    {timing.get('reasoning_agent_sec', 0):.3f} s",
        "",
        f"  total:             {timing.get('total_sec', 0):.3f} s",
        "",
        "=" * 60,
    ])
    (output_dir / f"{prefix}_timing.txt").write_text("\n".join(lines), encoding="utf-8")

    # 2. Append to timing_summary.csv
    spec_times = timing.get("specialists_sec", [])
    spec1 = f"{spec_times[0].get('sec', 0):.3f}" if len(spec_times) > 0 else ""
    spec2 = f"{spec_times[1].get('sec', 0):.3f}" if len(spec_times) > 1 else ""
    spec3 = f"{spec_times[2].get('sec', 0):.3f}" if len(spec_times) > 2 else ""
    row = [
        str(n),
        category,
        f"{timing.get('head_agent_sec', 0):.3f}",
        f"{timing.get('object_extraction_sec', 0):.3f}",
        spec1,
        spec2,
        spec3,
        f"{timing.get('reasoning_agent_sec', 0):.3f}",
        f"{timing.get('total_sec', 0):.3f}",
    ]
    csv_path = output_dir / "timing_summary.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "step,category,head_agent_sec,object_extraction_sec,specialist1_sec,specialist2_sec,specialist3_sec,reasoning_agent_sec,total_sec\n",
            encoding="utf-8",
        )
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")


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
    timing = {"head_agent_sec": 0.0, "object_extraction_sec": 0.0, "specialists_sec": [], "reasoning_agent_sec": 0.0}
    if answer_type is None:
        answer_type = _infer_answer_type(query)

    # 1. Head Agent -> category inference (or use force_category)
    head_raw = None
    if force_category and force_category in ALL_CATEGORIES:
        category = force_category
    else:
        _t = time.time()
        head_prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
        head_raw = head_generate(image, head_prompt)
        category = parse_category(head_raw, ALL_CATEGORIES)
        timing["head_agent_sec"] = round(time.time() - _t, 3)

    # 2. Agent selection from score map
    assignments = score_map.select_agents(category, step)

    # 2.5 Precompute object_names once when shared (saves ~1 VLM call)
    object_names_cache = None
    if shared_object_extraction:
        for role, llm_name in assignments:
            if role in _ROLES_WITH_TOOLS:
                _t = time.time()
                from src2.tools.object_extraction import extract_objects_from_image
                object_names_cache = extract_objects_from_image(image, specialist_generate, llm_name)
                timing["object_extraction_sec"] = round(time.time() - _t, 3)
                break

    # 3. Run 3 specialist agents -> SharedMemory (C: tools for explicit_3d, scene_graph)
    shared_memory = SharedMemory()
    agent_details = []
    tool_output_cache = {}  # role -> tool output (computed once per role type)
    for role, llm_name in assignments:
        _t_spec = time.time()
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
        timing["specialists_sec"].append({
            "role": role, "llm": llm_name,
            "sec": round(time.time() - _t_spec, 3),
        })
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
    _t = time.time()
    reasoning_prompt = build_final_reasoning_prompt(
        query, shared_memory.to_prompt_text(), with_image=use_vlm_reasoning, answer_type=answer_type
    )
    reasoning_raw = reasoning_generate(reasoning_prompt, image=image)
    timing["reasoning_agent_sec"] = round(time.time() - _t, 3)
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
    timing["total_sec"] = round(elapsed, 3)
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
        "timing": timing,
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
        _save_timing_to_file(Path(save_step_dir), step, category, timing, assignments)
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
