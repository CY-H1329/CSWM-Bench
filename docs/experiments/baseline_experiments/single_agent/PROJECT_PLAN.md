# Multi-Agent System for Spatial Reasoning Task

**Domaine d'étude** : Multi-Agent System for Spatial Reasoning Task

---

## Spatial Reasoning Tasks (9)

Depth, Distance, Relation, Existence, Count, Instance_location, Orientation, Size, Reach

---

## Benchmarks (4)

| Benchmark | Description |
|-----------|-------------|
| OMNI3D-BENCH | |
| CV-Bench | |
| 3DSRBench | |
| STVQA-7K | (déjà intégré dans le projet) |

---

## Architecture – Step 1

```
Input: Query + 2D Image
       ↓
   Head-Agent     → Analyse le query, classe le task (Depth, Relation, etc.)
       ↓
   Perception Agent → Décide comment résoudre:
                      • Direct (répondre sans extraction)
                      • Tools (3D: localisation, depth, taille, distance)
                      • Mask / Bounding box, etc.
       ↓
   Reasoning Agent  → Raisonne avec les infos extraites
                      (ou 1 seul agent si réponse rapide sans CoT)
       ↓
Output: Answer
```

**Process** : `input → Head-Agent → Perception Agent → Reasoning Agent → answer`

---

## Modèles à tester

| Modèle | Rôle(s) / Notes |
|--------|------------------|
| Qwen-3.0-VL 4B/8B | Open source |
| Llava-4D | Open source |
| Sa2VA | SpatialReasoner-R1 backbone, open source |
| Claude 3.5 V Sonnet | API |
| GPT-4o | API |
| Deepseek | Head: deepseek-v3.2 (chat) ; Perception: deepseek-vl2 (multimodal) ; Reasoning: deepseek-r1 (reasoner) |
| Gemini 2.5 flash-Lite | API |
| Gemini Robotics-ER 1.5 | Embodied Reasoning, 3D coords / trajectoires |

---

## Plan de tests – Step 1

### Phase 1 : Open source sur 4 benchmarks

Modèles : **Qwen-3.0 4B**, **Llava-4D**, **Sa2VA**

**1.1 Agents identiques**
- Head = Perception = Reasoning = Qwen-3.0 4B
- Head = Perception = Reasoning = Llava-4D
- Head = Perception = Reasoning = Sa2VA

**1.2 Toutes les combinaisons (3 modèles)**
- Head / Perception / Reasoning = C(3,1) × C(3,1) × C(3,1) = **27 combinaisons**

### Phase 2 : Modèles cloud / API

- Claude 3.5 V Sonnet, GPT-4o
- Deepseek (Head / Perception / Reasoning comme défini)
- Gemini 2.5 flash-Lite, Gemini Robotics-ER 1.5

---

## Organisation proposée

```
Spatial_MAS/
├── benchmarks/                    # Données par benchmark
│   ├── stvqa7k/
│   ├── omni3d_bench/
│   ├── cv_bench/
│   └── 3dsrbench/
├── agents/                        # Implémentation des agents
│   ├── head_agent.py
│   ├── perception_agent.py
│   └── reasoning_agent.py
├── runs/                          # Résultats par benchmark × combinaison
│   ├── stvqa7k/
│   │   ├── qwen4b_qwen4b_qwen4b/
│   │   ├── llava4d_llava4d_llava4d/
│   │   ├── sa2va_sa2va_sa2va/
│   │   ├── qwen4b_llava4d_sa2va/
│   │   └── ...
│   ├── omni3d_bench/
│   │   └── ...
│   └── ...
├── configs/                       # Config par combinaison
│   └── step1_phase1.yaml
└── docs/experiments/
    ├── PROJECT_PLAN.md           # Ce document
    └── ...
```

---

## Implémentation (Step 1)

### Setup H100 (une fois)

```bash
python scripts/setup_datasets.py   # Télécharge les 4 benchmarks
```

### Structure du code

- `src/benchmarks/` : loaders pour STVQA-7K, OMNI3D, CV-Bench, 3DSRBench
- `src/agents/` : pipeline Head → Perception → Reasoning
- `run_eval_mas.py` : évaluation MAS

### Head-Agent

**Ne reçoit PAS la catégorie** : il doit classifier le task lui-même à partir de l’image + query.

### Exécution

```bash
# Agents identiques (Qwen)
python run_eval_mas.py --benchmark stvqa7k --head qwen --perception qwen --reasoning qwen

# Combinaison mixte
python run_eval_mas.py --benchmark stvqa7k --head qwen --perception llava --reasoning qwen

# H100
bash scripts/run_h100_mas.sh stvqa7k qwen qwen qwen --max_per_category 10
```

### Résultats

`results/runs/<benchmark>/<head>_<perception>_<reasoning>/<timestamp>/`

---

## Step 2 & Step 3

À définir.
