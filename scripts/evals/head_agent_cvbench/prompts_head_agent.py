"""
Head-Agent 5가지 핵심 능력 프롬프트.

1. Task Decomposition - 문제 정확 분류 (잘못 분류 시 trust 학습 왜곡)
2. Routing Decision - 어떤 agent 3명 중 누구를 고를지 (specialization 학습의 시작점)
3. Complexity Estimation - 간단 vs 복잡 판단 (tool 과용 방지 / shortcut 방지)
4. Strategy Planning - tool/strategy 초안 제시 (Perception policy의 출발점)
5. Trust-Aware Logging - reasoning trace 구조화 (추후 trust update 신호 확보)
"""
from typing import List
from prompt import get_categories


# === 1. Task Decomposition (기존 category routing과 동일) ===
def build_task_decomposition_prompt(question: str, benchmark: str) -> str:
    """문제를 정확히 분류. 잘못 분류하면 trust 학습이 왜곡됨."""
    categories = get_categories(benchmark)
    cats = "\n".join(f"- {c}" for c in categories)
    return f"""# TASK DECOMPOSITION (Head-Agent Capability 1)

You are the Head-Agent. Your first job is to **accurately classify** the problem. Wrong classification distorts downstream trust learning.

## Benchmark: {benchmark.upper()}
Available task categories:
{cats}

## Instructions
1. Read the question carefully.
2. Select the ONE category that best matches what the question is asking.
3. Use the exact category name from the list.

## Output format
Category: <your selected category>

---

## Question
{question}
"""


# === 2. Routing Decision ===
ROUTING_OPTIONS = ["Direct", "Perception", "Reasoning", "Both"]


def build_routing_decision_prompt(question: str, category: str, benchmark: str) -> str:
    """어떤 agent를 고를지. specialization 학습의 시작점."""
    opts = "\n".join(f"- {o}" for o in ROUTING_OPTIONS)
    return f"""# ROUTING DECISION (Head-Agent Capability 2)

You are the Head-Agent. Given the question and its task category, decide **which agent(s)** should handle it.

Task category (from decomposition): {category}
Benchmark: {benchmark.upper()}

## Agent options
{opts}

- **Direct**: Answer immediately without extraction (simple lookup, obvious visual answer)
- **Perception**: Needs visual extraction (depth, localization, object detection, counting)
- **Reasoning**: Needs multi-step spatial reasoning (relations, comparisons, ordering)
- **Both**: Needs both Perception (extraction) AND Reasoning (multi-step)

## Instructions
1. Consider what the question requires.
2. Choose exactly ONE option.
3. Brief reasoning (1 sentence) then your decision.

## Output format
Reasoning: <1 sentence>
Route: <Direct|Perception|Reasoning|Both>

---

## Question
{question}
"""


# === 3. Complexity Estimation ===
COMPLEXITY_LEVELS = ["1", "2", "3", "4", "5"]  # 1=simple, 5=complex


def build_complexity_estimation_prompt(question: str, category: str) -> str:
    """간단 vs 복잡 판단. tool 과용 방지 / shortcut 방지."""
    return f"""# COMPLEXITY ESTIMATION (Head-Agent Capability 3)

You are the Head-Agent. Estimate the **complexity** of this task. This prevents tool overuse and inappropriate shortcuts.

Task category: {category}

## Scale (1-5)
- 1: Trivial (single object, immediate answer)
- 2: Simple (1-2 steps, minimal reasoning)
- 3: Moderate (multi-step, some comparison)
- 4: Complex (multiple objects, spatial relations, ordering)
- 5: Very complex (multi-hop reasoning, fine-grained comparison)

## Instructions
1. Read the question.
2. Assign a complexity score 1-5.
3. One sentence justification.

## Output format
Complexity: <1|2|3|4|5>
Justification: <1 sentence>

---

## Question
{question}
"""


# === 4. Strategy Planning ===
def build_strategy_planning_prompt(question: str, category: str, route: str) -> str:
    """tool/strategy 초안 제시. Perception policy의 출발점."""
    return f"""# STRATEGY PLANNING (Head-Agent Capability 4)

You are the Head-Agent. Propose an **initial strategy** for solving this task. This is the starting point for Perception policy.

Task category: {category}
Routing: {route}

## Possible tools/strategies (examples)
- Depth estimation, 3D localization
- Object detection, counting, segmentation
- Spatial relation extraction (above, left, closer)
- Multi-object comparison, ordering

## Instructions
1. List 1-3 concrete steps or tools needed.
2. Be specific to this question.
3. Keep it brief (2-4 sentences).

## Output format
Strategy:
1. <step or tool>
2. <step or tool>
3. <step or tool> (optional)

---

## Question
{question}
"""


# === 5. Trust-Aware Logging ===
def build_trust_aware_logging_prompt(question: str, category: str, route: str, complexity: str) -> str:
    """reasoning trace를 구조화. 추후 trust update 신호 확보."""
    return f"""# TRUST-AWARE LOGGING (Head-Agent Capability 5)

You are the Head-Agent. Produce a **structured reasoning trace** for downstream trust updates.

Task category: {category}
Routing: {route}
Complexity: {complexity}

## Instructions
1. Summarize your reasoning in a structured way.
2. Output valid JSON with the exact keys below.
3. Keep it concise but complete.

## Output format (JSON only)
{{
  "reasoning": "<2-3 sentence summary of why you made these decisions>",
  "category": "<task category>",
  "route": "<Direct|Perception|Reasoning|Both>",
  "complexity": "<1-5>",
  "confidence": <0.0-1.0>,
  "key_factors": ["<factor1>", "<factor2>"]
}}

---

## Question
{question}
"""
