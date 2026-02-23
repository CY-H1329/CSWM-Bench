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


def parse_specialist_output(raw: str) -> Tuple[str, str]:
    """Extract (answer, reason) from specialist output."""
    raw = (raw or "").strip()
    answer = ""
    reason = ""

    ans_m = re.search(
        r"Answer\s*:\s*\(?([A-D])\)?",
        raw, re.IGNORECASE,
    )
    if ans_m:
        answer = f"({ans_m.group(1).upper()})"

    reason_m = re.search(
        r"Reason\s*:\s*(.+?)(?=\nAnswer\s*:|\Z)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    if reason_m:
        reason = reason_m.group(1).strip()[:2000]

    if not answer:
        fallback = re.search(r"\(([A-D])\)", raw)
        if fallback:
            answer = f"({fallback.group(1)})"

    return answer, reason


def parse_final_answer(raw: str) -> str:
    """Extract the final answer letter from Reasoning Agent output."""
    raw = (raw or "").strip()
    m = re.search(r"Answer\s*:\s*\(?([A-D])\)?", raw, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    m = re.search(r"\(([A-D])\)", raw)
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
    reasoning_generate: Callable[[str], str],
    updater: Optional[ScoreMapUpdater] = None,
    update_scores: bool = True,
) -> Dict:
    """Execute one step of the MAS v2 pipeline.

    Categories are always the fixed ALL_CATEGORIES (16 types).
    The Head Agent classifies any question into the best-fit category.
    """
    t0 = time.time()

    # 1. Head Agent -> category inference (Qwen3-VL-4B, image + text)
    head_prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
    head_raw = head_generate(image, head_prompt)
    category = parse_category(head_raw, ALL_CATEGORIES)

    # 2. Agent selection from score map
    assignments = score_map.select_agents(category, step)

    # 3. Run 3 specialist agents -> SharedMemory (C: tools for explicit_3d, scene_graph)
    shared_memory = SharedMemory()
    agent_details = []
    tool_output_cache = {}  # role -> tool output (computed once per role type)
    for role, llm_name in assignments:
        if role in _ROLES_WITH_TOOLS and role not in tool_output_cache:
            try:
                if role == "explicit_3d_representation":
                    from src2.tools import get_depth_summary
                    tool_output_cache[role] = get_depth_summary(image)
                elif role == "scene_graph_construction":
                    from src2.tools import get_scene_graph_summary
                    tool_output_cache[role] = get_scene_graph_summary(image)
            except Exception as e:
                logger.warning("Tool for %s failed: %s", role, e)
                tool_output_cache[role] = ""
        tool_output = tool_output_cache.get(role, None)
        role_prompt = build_role_prompt(role, query, tool_output=tool_output)
        raw_output = specialist_generate(llm_name, image, role_prompt)
        answer, reason = parse_specialist_output(raw_output)
        shared_memory.add(role, llm_name, answer, reason)
        agent_details.append({
            "role": role,
            "llm_name": llm_name,
            "answer": answer,
            "reason": reason,
            "raw_output": raw_output[:3000],
        })

    # 4. Final Reasoning Agent (DeepSeek-R1, text-only: query + SharedMemory)
    reasoning_prompt = build_final_reasoning_prompt(query, shared_memory.to_prompt_text())
    reasoning_raw = reasoning_generate(reasoning_prompt)
    final_answer = parse_final_answer(reasoning_raw)

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
        "correct": _norm_letter(final_answer) == _norm_letter(gt) if gt else None,
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
) -> List[Dict]:
    """Test phase: iterate over dataset, score map is frozen (no updates).

    Categories are always the fixed ALL_CATEGORIES (16 types).
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

        result = run_step(
            image=image,
            query=query,
            gt=gt,
            step=step + 999999,
            total_steps=total,
            score_map=score_map,
            head_generate=head_generate,
            specialist_generate=specialist_generate,
            reasoning_generate=reasoning_generate,
            updater=None,
            update_scores=False,
        )
        results.append(result)

        if (step + 1) % 50 == 0 or step == total - 1:
            correct_so_far = sum(1 for r in results if r.get("correct"))
            logger.info(
                "Test step %d/%d | running acc: %.2f%%",
                step + 1, total,
                100.0 * correct_so_far / len(results),
            )

    return results


# ======================================================================
# Helpers
# ======================================================================
def _norm_letter(s: str) -> str:
    s = (s or "").strip().upper()
    for c in "ABCD":
        if c in s:
            return c
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
