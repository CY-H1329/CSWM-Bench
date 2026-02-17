"""
Shared logic for GQA single-model evaluation scripts.
GQA categories: structural (query, verify, choose, logical, compare),
semantic (relation, attribute, object, global, other).
"""
from typing import Optional

# GQA structural types
GQA_STRUCTURAL_CATEGORIES = [
    "query",
    "verify",
    "choose",
    "logical",
    "compare",
]

# GQA semantic types
GQA_SEMANTIC_CATEGORIES = [
    "relation",
    "attribute",
    "object",
    "global",
    "other",
]


def build_spatial_prompt(question: str, task_category: Optional[str] = None) -> str:
    """
    Build the spatial/reasoning prompt for GQA.
    Agent infers task type in STEP 1.
    """
    cats = "\n".join(f"- {c}" for c in GQA_SEMANTIC_CATEGORIES)
    return f"""# ROLE
You are an expert in visual reasoning and compositional question answering.
Your objective is to answer accurately based on the image and question.

---

# INPUT
You will receive:
- An image
- A question

---

# STEP 1 — TASK CLASSIFICATION

Classify the question into exactly ONE of the following semantic types:

{cats}

Rules:
- Select only one type.
- Choose the most dominant reasoning type required.
- Do not skip this step.

---

# STEP 2 — REASONING

Based on the selected type:
1. Identify relevant visual cues.
2. Apply logical reasoning.
3. Avoid guessing.

---

# STEP 3 — FINAL ANSWER

Provide a concise final answer. For yes/no: "yes" or "no". For others: the specific answer (e.g., color, number, object name).

---

# OUTPUT FORMAT

Task Type:
<One of the semantic types above>

Reasoning:
<Brief reasoning steps>

Final Answer:
<Your answer>

---

# QUESTION

{question}
"""
