"""
Shared logic for CV-Bench single-model evaluation scripts.
CV-Bench: 2D (spatial relationships, object counting) + 3D (depth order, relative distance).
Agent infers task category by itself.
"""
from typing import Optional

# CV-Bench task categories (from explore_categories.py)
# 2D: Count, Relation | 3D: Depth, Distance
CVBENCH_TASK_CATEGORIES = [
    "Count",
    "Relation",
    "Depth",
    "Distance",
]


def build_spatial_prompt(question: str, task_category: Optional[str] = None) -> str:
    """
    Build the spatial/vision reasoning prompt for CV-Bench.
    Category is NOT given — agent must infer it.
    """
    cats = "\n".join(f"- {c}" for c in CVBENCH_TASK_CATEGORIES)
    return f"""# ROLE
You are an expert in visual and spatial reasoning.
Your objective is to solve vision-centric tasks accurately: 2D understanding (spatial relationships, object counting) and 3D understanding (depth order, relative distance).

---

# INPUT
You will receive:
- An image
- A question with multiple choice options

---

# STEP 1 — TASK CLASSIFICATION

Classify the question into exactly ONE of the following categories:

{cats}

Rules:
- Select only one category.
- If multiple seem relevant, choose the most dominant reasoning type required.
- Do not skip this step.

---

# STEP 2 — TASK-SPECIFIC PLAN

Based on the selected category:

1. Define the key visual cues needed (spatial layout, occlusion, counting, depth, distance).
2. Identify relevant visual features.
3. Explain your strategy to solve this specific task.
4. Avoid superficial shortcuts or guessing.

---

# STEP 3 — STEP-BY-STEP REASONING

Follow a strict logical reasoning process:

- Analyze the image carefully.
- Extract relevant visual and spatial information.
- Apply geometric or spatial logic when necessary.
- Ensure each reasoning step follows logically from the previous one.
- Do NOT jump directly to the answer.

---

# STEP 4 — FINAL ANSWER

Provide:
- A concise final answer.
- For multiple choice: reply with **Final Answer: (X)** where X is A, B, C, or D.

---

# OUTPUT FORMAT (MANDATORY)

Task Category:
<One of the 4 categories>

Reasoning Plan:
<Brief task-specific plan>

Step-by-Step Reasoning:
<Logical reasoning steps>

Final Answer:
(Answer letter in parentheses, e.g. (A), (B), (C), or (D))

---

# QUESTION

{question}
"""
