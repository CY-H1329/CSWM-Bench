"""
Shared logic for 3DSRBench single-model evaluation scripts.
Agent infers task category by itself (12 fine-grained categories).
"""
from typing import Optional

# 3DSRBench task categories (fine-grained)
DSR3BENCH_TASK_CATEGORIES = [
    "location_above",
    "height_higher",
    "location_closer_to_camera",
    "multi_object_closer_to",
    "orientation_on_the_left",
    "multi_object_facing",
    "multi_object_same_direction",
    "orientation_in_front_of",
    "multi_object_viewpoint_towards_object",
    "orientation_viewpoint",
    "location_next_to",
    "multi_object_parallel",
]

# CV-Bench task column (`task` on HF): Count, Relation, Depth, Distance
CV_BENCH_TASK_CATEGORIES = ["Count", "Relation", "Depth", "Distance"]


def build_spatial_prompt(question: str, task_category: Optional[str] = None) -> str:
    """
    Build the spatial reasoning prompt.
    Category is NOT given — agent must infer it in STEP 1.
    """
    cats = "\n".join(f"- {c}" for c in DSR3BENCH_TASK_CATEGORIES)
    return f"""# ROLE
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

{cats}

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
- For multiple choice (A/B/C/D): reply with **Final Answer: (X)** where X is A, B, C, or D.

---

# OUTPUT FORMAT (MANDATORY)

Task Category:
<One of the 12 categories>

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


def build_cvbench_prompt(question: str, task_category: Optional[str] = None) -> str:
    """CV-Bench MCQ prompt — model infers task type (4 categories), analogous to 3DSRBench."""
    cats = "\n".join(f"- {c}" for c in CV_BENCH_TASK_CATEGORIES)
    return f"""# ROLE
You are an expert in computer vision and spatial reasoning (CV-Bench style).
Your objective is to answer multiple-choice questions about the image accurately.

---

# INPUT
You will receive:
- An image
- A question with options A/B/C/D

---

# STEP 1 — TASK CLASSIFICATION

Classify the question into exactly ONE of the following CV-Bench task types:

{cats}

Rules:
- Select only one type.
- Choose the type that best describes what the question asks (counting, spatial relation, depth ordering, distance).
- Do not skip this step.

---

# STEP 2 — TASK-SPECIFIC PLAN

Based on the selected type, outline briefly how you will use the image to decide.

---

# STEP 3 — STEP-BY-STEP REASONING

Analyze the image and reason step by step. Do not guess without looking.

---

# STEP 4 — FINAL ANSWER

For multiple choice: reply with **Final Answer: (X)** where X is A, B, C, or D.

---

# OUTPUT FORMAT (MANDATORY)

Task Category:
<One of: Count, Relation, Depth, Distance>

Reasoning Plan:
<Brief plan>

Step-by-Step Reasoning:
<Steps>

Final Answer:
(Letter in parentheses, e.g. (A), (B), (C), or (D))

---

# QUESTION

{question}
"""
