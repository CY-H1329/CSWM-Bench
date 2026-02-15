"""
Shared logic for 3DSRBench single-model evaluation scripts.
Agent infers task category by itself (Height, Location, Orientation, Multi-Object).
"""
from typing import Optional

# 3DSRBench official task categories — agent must classify by itself
DSR3BENCH_TASK_CATEGORIES = ["Height", "Location", "Orientation", "Multi-Object"]


def build_spatial_prompt(question: str, task_category: Optional[str] = None) -> str:
    """
    Build the spatial reasoning prompt.
    Category is NOT given — agent must infer it in STEP 1.
    """
    return """# ROLE
You are an expert in spatial reasoning.
Your objective is to solve visual spatial reasoning tasks accurately and systematically.

---

# INPUT
You will receive:
- An image
- A question

---

# STEP 1 — TASK CLASSIFICATION

Classify the question into exactly ONE of the following categories:

- Height
- Location
- Orientation
- Multi-Object

Rules:
- Select only one category.
- If multiple seem relevant, choose the most dominant spatial reasoning type required to answer correctly.
- Do not skip this step.

---

# STEP 2 — TASK-SPECIFIC PLAN

Based on the selected category:

1. Define the key spatial cues needed.
2. Identify relevant visual features (e.g., occlusion, perspective, alignment, relative scale).
3. Explain your strategy to solve this specific task.
4. Avoid superficial shortcuts or guessing.

---

# STEP 3 — STEP-BY-STEP REASONING

Follow a strict logical reasoning process:

- Analyze the image carefully.
- Extract relevant spatial information.
- Apply geometric or spatial logic when necessary.
- Ensure each reasoning step follows logically from the previous one.
- Do NOT jump directly to the answer.

---

# STEP 4 — FINAL ANSWER

Provide:
- A concise final answer.
- If multiple choices exist, clearly indicate the selected option.

---

# OUTPUT FORMAT (MANDATORY)

Task Category:
<One of the 4 categories>

Reasoning Plan:
<Brief task-specific plan>

Step-by-Step Reasoning:
<Logical reasoning steps>

Final Answer:
<Clear final answer>

---

# QUESTION

""" + question
