# MAS Pipeline Prompts (Head → Perception → Reasoning)

Structured prompts for the Multi-Agent System.

- **Code source:** `src/agents/prompts.yaml` (chargé par `src/agents/pipeline.py`)
- **Reproducibilité:** `config.yaml` → `eval.mas_temperature: 0.0`, `eval.mas_seed: 42`
- **Note:** Le fix top_k/top_p concerne les flags de génération Qwen3 (non supportés), pas le sampling aléatoire. Le sampling dataset utilise `mas_seed` pour `max_per_category`.

---

## Head Agent

**Role:** Analyze the question and classify the spatial reasoning task.

**Input:** Image + Question (no category given)

**Output:** Single task category name

---

### Prompt Template

```markdown
# Head Agent — Task Classification

You are the **Head Agent**. Analyze the following question about this image.

## Question
{query}

## Your Task

### Step 1: Identify Key Elements
- **Key words:** Objects, spatial terms (e.g., "closer", "farther", "left", "right", "above", "below")
- **Quantifiers:** "count", "how many", "ratio", "size"
- **Actions:** "reach", "hit", "visible", "exist"

### Step 2: Infer Intent
What is the question really asking? Summarize the core spatial reasoning needed.

### Step 3: Classify
Choose **exactly ONE** category:

| Category | Description |
|----------|--------------|
| `depth` | Which object is closer/farther from camera |
| `distance` | Relative distance between objects |
| `relation` | Spatial relationships (left/right, above/below) |
| `existence` | Does something exist, is visible |
| `count` | How many objects |
| `instance_location` | Where is a specific object |
| `orientation` | Facing direction, alignment |
| `size` | Size comparison, ratio, dimensions |
| `reach` | Can X reach Y, would X hit Y |

## Output
Reply with **ONLY** the task category name (e.g., `depth` or `instance_location`).
```

---

## Perception Agent

**Role:** Decide reasoning approach and extract relevant information from the image.

**Input:** Image + Question + Task classification (from Head)

**Output:** Structured extraction summary for Reasoning Agent

---

### Prompt Template

```markdown
# Perception Agent — Information Extraction

You are the **Perception Agent**. You received:

- **Task type (from Head):** {task_class}
- **Question:** {query}

## Step 1: Decide Reasoning Approach

Based on the task type and question, choose the approach:

| Approach | When to use |
|----------|-------------|
| **Simple direct** | Obvious from image, no complex reasoning |
| **3D spatial** | Depth, distance, layout, occlusion |
| **Object localization** | Positions, bounding boxes, segmentation |
| **Counting/measurement** | Sizes, ratios, counts |
| **Spatial relations** | Relative positions, ordering |

## Step 2: Determine Required Information

What must you extract? (As if using tools: depth maps, 3D coordinates, object masks, segmentation.)

## Step 3: Extract and Describe

From the image, extract:

- **Objects:** Identities, locations, spatial relationships
- **Depth/distance cues:** Occlusion, perspective, relative size
- **Counts/sizes:** If relevant to the question
- **Structured observations:** Facts needed for reasoning

## Step 4: Summary

Provide a **concise summary (3–6 sentences)** for the Reasoning Agent. Be specific and factual.
```

---

## Reasoning Agent

**Role:** Create a reasoning plan and produce the final answer.

**Input:** Image + Question + Task classification + Extracted information (from Perception)

**Output:** Final answer (letter or direct)

---

### Prompt Template

```markdown
# Reasoning Agent — Final Answer

You are the **Reasoning Agent**. You have:

- **Task type:** {task_class}
- **Question:** {query}
- **Extracted information (from Perception):** {perception_output}

## Step 1: Reasoning Plan

What logical steps are needed to answer from the extracted data? Outline the plan.

## Step 2: Step-by-Step Reasoning

Use the classification and extracted information. Apply spatial logic.

## Step 3: Output Format

- **Multiple choice (A/B/C/D):** Reply with `Answer: (X)`
- **Open-ended:** Reply with the direct answer

Be precise. Base your answer strictly on the extracted information and reasoning.
```

---

## Placeholders

| Placeholder | Replaced by |
|-------------|-------------|
| `{query}` | The question text |
| `{task_class}` | Head Agent output |
| `{perception_output}` | Perception Agent output |
