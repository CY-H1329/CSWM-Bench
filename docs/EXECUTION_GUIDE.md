# Guide d'exécution — Scripts organisés par rôle

Ce document décrit tous les scripts d'évaluation, leur rôle, les commandes et les prompts utilisés.

---

## Vue d'ensemble des scripts

| Rôle | Script | Benchmark(s) | Description |
|------|--------|--------------|-------------|
| **Single-agent** | `run_eval.py` | STVQA-7K | Un modèle par échantillon, comparaison directe |
| **Multi-agent (vote)** | `run_eval_multiagent.py` | STVQA-7K | 3 agents identiques, vote majoritaire |
| **Unified** | `run_eval_unified.py` | STVQA-7K | Single + Multi en une exécution |
| **Collab** | `run_eval_collab.py` | STVQA-7K | Qwen + LLaVA en collaboration |
| **MAS (pipeline)** | `run_eval_mas.py` | 4 benchmarks | Head → Perception → Reasoning (1 combinaison) |
| **MAS Full** | `run_eval_mas_full.py` | 4 benchmarks | Qwen3/Sa2VA/LLaVA4D ×3, données complètes |
| **Single 3DSRBench** | `run_eval_single_3dsrbench.py` | 3DSRBench | Qwen3, Sa2VA, LLaVA4D (tous en une fois) |
| **3DSRBench Qwen3** | `scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py` | 3DSRBench | Qwen3-4B uniquement (recommandé) |
| **3DSRBench Sa2VA** | `scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py` | 3DSRBench | Sa2VA uniquement (recommandé) |
| **3DSRBench LLaVA4D** | `scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py` | 3DSRBench | LLaVA4D uniquement (recommandé) |

---

## 1. Single-agent (`run_eval.py`)

**Rôle** : Évaluer un ou plusieurs modèles individuellement sur STVQA-7K.

```bash
python run_eval.py --models qwen llava --split val
python run_eval.py --models qwen3_4b sa2va llava4d --split val --max_samples 100
```

**Prompt** : Question + options (format standard du benchmark). Pas de prompt de raisonnement structuré.

---

## 2. Multi-agent vote (`run_eval_multiagent.py`)

**Rôle** : 3 instances du même modèle, réponse par vote majoritaire.

```bash
python run_eval_multiagent.py --models qwen llava --split val --max_per_category 50
```

---

## 3. Unified (`run_eval_unified.py`)

**Rôle** : Exécuter Single puis Multi en une seule commande.

```bash
python run_eval_unified.py --models qwen llava --split train --max_per_category 100
```

---

## 4. Collab (`run_eval_collab.py`)

**Rôle** : Qwen + LLaVA en mode collaboration (discussion, consensus).

```bash
python run_eval_collab.py --split train --max_per_category 50
```

---

## 5. MAS Pipeline (`run_eval_mas.py`)

**Rôle** : Pipeline Head → Perception → Reasoning. Une combinaison de modèles.

```bash
# Qwen3 pour les 3 agents
python run_eval_mas.py --benchmark stvqa7k --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b

# Sa2VA
python run_eval_mas.py --benchmark stvqa7k --head sa2va --perception sa2va --reasoning sa2va

# LLaVA4D
python run_eval_mas.py --benchmark stvqa7k --head llava4d --perception llava4d --reasoning llava4d

# Test rapide
python run_eval_mas.py --benchmark stvqa7k --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b --max_samples 100 --seed 123
```

**Benchmarks** : `stvqa7k`, `omni3d`, `cvbench`, `3dsrbench`

### Prompts MAS (src/agents/prompts.yaml)

#### Head Agent
```
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
- depth, distance, relation, existence, count
- instance_location, orientation, size, reach

## Output
Reply with **ONLY** the task category name (e.g., depth or instance_location).
```

#### Perception Agent
```
# Perception Agent — Information Extraction

You are the **Perception Agent**. You received:
- **Task type (from Head):** {task_class}
- **Question:** {query}

## Step 1: Decide Reasoning Approach
Based on the task type and question, choose: Simple direct | 3D spatial | Object localization | Counting/measurement | Spatial relations.

## Step 2: Determine Required Information
What must you extract? (As if using tools: depth maps, 3D coordinates, object masks, segmentation.)

## Step 3: Extract and Describe
From the image: object identities, locations, spatial relationships; depth/distance cues (occlusion, perspective); counts/sizes if relevant.

## Step 4: Summary
Provide a **concise summary (3–6 sentences)** for the Reasoning Agent. Be specific and factual.
```

#### Reasoning Agent
```
# Reasoning Agent — Final Answer

You are the **Reasoning Agent**. You have:
- **Task type:** {task_class}
- **Question:** {query}
- **Extracted information (from Perception):** {perception_output}

## Step 1: Reasoning Plan
What logical steps are needed to answer from the extracted data?

## Step 2: Step-by-Step Reasoning
Use the classification and extracted information. Apply spatial logic.

## Step 3: Output Format
- Multiple choice (A/B/C/D): Reply with "Answer: (X)"
- Open-ended: Reply with the direct answer

Be precise. Base your answer strictly on the extracted information and reasoning.
```

---

## 6. MAS Full (`run_eval_mas_full.py`)

**Rôle** : Exécuter les 3 combinaisons (Qwen3×3, Sa2VA×3, LLaVA4D×3) sur toutes les données. Résultats par catégorie, step_outputs sauvegardés.

```bash
python run_eval_mas_full.py --benchmark stvqa7k
python run_eval_mas_full.py --benchmark 3dsrbench --seed 42
```

**Sorties** : `results/runs/<benchmark>/full_eval/<timestamp>/` avec `by_category_summary.txt`, `step_outputs/sample_*.txt`, `all_combinations_summary.txt`.

---

## 7. 3DSRBench — Scripts par modèle (recommandé)

**Rôle** : Exécuter chaque modèle **séparément** pour éviter toute interférence (résultats identiques, fuite mémoire). L'agent **infère lui-même** la catégorie (Height, Location, Orientation, Multi-Object).

```bash
# Qwen3-4B uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py
python scripts/evals/3dsrbench/run_eval_3dsrbench_qwen3.py --max_samples 50 --seed 42

# Sa2VA uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_sa2va.py

# LLaVA4D uniquement
python scripts/evals/3dsrbench/run_eval_3dsrbench_llava4d.py
```

**Sorties** : `results/runs/3dsrbench/<model>/<timestamp>/` avec `responses/sample_*.txt`, `details.jsonl`, `results.json`.

### 7.1. Single 3DSRBench (tous modèles en une fois)

```bash
python run_eval_single_3dsrbench.py
python run_eval_single_3dsrbench.py --max_samples 50 --seed 42
```

### Prompt 3DSRBench (raisonnement spatial)

L'agent **infère la catégorie** en STEP 1 (Height, Location, Orientation, Multi-Object). La catégorie n'est pas fournie.

```
# ROLE
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

{question}
```

---

## Benchmarks supportés

| Benchmark | Clé | Split | Options |
|-----------|-----|-------|---------|
| STVQA-7K | `stvqa7k` | val | Oui (A/B/C/D) |
| Omni3D-Bench | `omni3d` | train | Non |
| CV-Bench | `cvbench` | test | Oui |
| 3DSRBench | `3dsrbench` | test | Oui (A/B/C/D) |

---

## Structure des résultats

```
results/
├── runs/
│   ├── stvqa7k/
│   │   ├── full_eval/<timestamp>/     # MAS Full
│   │   │   ├── qwen3_4b_qwen3_4b_qwen3_4b/
│   │   │   ├── sa2va_sa2va_sa2va/
│   │   │   ├── llava4d_llava4d_llava4d/
│   │   │   └── all_combinations_summary.txt
│   │   └── <head>_<perc>_<reas>/<timestamp>/  # MAS single combo
│   └── 3dsrbench/
│       ├── qwen3_4b/<timestamp>/      # Script séparé Qwen3
│       ├── sa2va/<timestamp>/        # Script séparé Sa2VA
│       ├── llava4d/<timestamp>/      # Script séparé LLaVA4D
│       └── single_eval/<timestamp>/  # run_eval_single_3dsrbench (tous)
└── YYYYMMDD_HHMMSS/                   # run_eval.py, etc.
```

---

## Configuration (config.yaml)

- `models.qwen3_4b`, `models.sa2va`, `models.llava4d` : model_id, device
- `eval.mas_temperature` : 0.0 (greedy) pour reproductibilité
- `eval.mas_seed` : seed pour l'échantillonnage
- `eval.max_new_tokens` : 512 (ou 1024 pour 3DSRBench single)
