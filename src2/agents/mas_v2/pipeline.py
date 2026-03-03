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
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from .config import SPECIALIST_LLMS, ROLES, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS
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
        agent_details.append({
            "role": role,
            "llm_name": llm_name,
            "answer": answer,
            "reason": reason,
            "raw_output": raw_output[:3000],
        })

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
    return {
        "step": step,
        "category": category,
        "assignments": [(r, l) for r, l in assignments],
        "agent_details": agent_details,
        "final_answer": final_answer,
        "gt": gt,
        "correct": _is_correct(final_answer, gt, answer_type) if gt else None,
        "reasoning_raw": reasoning_raw[:3000],
        "elapsed_sec": round(elapsed, 2),
    }


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
    updater: "ScoreMapUpdater" = None,
    update_scores: bool = False,
) -> List[Dict]:
    """Test phase: iterate over dataset.

    Categories are always the fixed ALL_CATEGORIES (16 types).
    random_agents: If True, use step=0 for each sample (random agent selection).
    verbose: If True, log step/acc/cat/assign and scores (every 5 steps).
    updater + update_scores: If set, update score map after each sample (TTO).
    """
    from src2.benchmarks.loaders import (
        get_benchmark_image, get_benchmark_prompt, get_benchmark_answer,
    )
    _img = get_image_fn or (lambda ex: get_benchmark_image(ex, benchmark))
    _prompt = get_prompt_fn or (lambda ex: get_benchmark_prompt(ex, benchmark))
    _answer = get_answer_fn or (lambda ex: get_benchmark_answer(ex, benchmark))

    total = len(dataset)
    results = []

    for step in range(total):
        example = dataset[step]
        image = _img(example)
        query = _prompt(example)
        gt = _answer(example)

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
        )
        results.append(result)

        correct_so_far = sum(1 for r in results if r.get("correct"))
        acc_pct = 100.0 * correct_so_far / len(results) if results else 0

        if verbose:
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


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute overall and per-category accuracy from result dicts."""
    total = 0
    correct = 0
    by_cat: Dict[str, Dict[str, int]] = {}

    for r in results:
        if r.get("correct") is None:
            continue
        total += 1
        if r["correct"]:
            correct += 1
        cat = r.get("category", "unknown")
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
