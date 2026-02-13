# Push / Pull — MAS (Qwen3-4B, Sa2VA, LLaVA4D)

---

## 1. Push (Local → GitHub)

```bash
cd ~/Desktop/Spatial_MAS

git add .
git status
git commit -m "Add Qwen3-4B, Sa2VA, LLaVA4D; MAS pipeline; 4 benchmarks"
git push origin main
```

---

## 2. Pull (H100 Server ← GitHub)

```bash
cd /path/to/Spatial_MAS   # ou ~/CY/Spatial_MAS

git pull origin main
conda activate spatial_mas
pip install -r requirements.txt
```

---

## 3. H100 — Setup & Run

```bash
# Datasets (une fois)
python scripts/setup_datasets.py

# MAS — agents identiques
bash scripts/run_h100_mas.sh stvqa7k qwen3_4b qwen3_4b qwen3_4b --max_per_category 10
bash scripts/run_h100_mas.sh stvqa7k llava4d llava4d llava4d --max_per_category 10
bash scripts/run_h100_mas.sh stvqa7k sa2va sa2va sa2va --max_per_category 10

# MAS — combinaisons
bash scripts/run_h100_mas.sh stvqa7k qwen3_4b llava4d sa2va
```

---

## Modèles configurés

| Alias     | Modèle                    | Note                    |
|-----------|---------------------------|-------------------------|
| qwen3_4b  | Qwen/Qwen3-VL-4B-Instruct | transformers>=4.51      |
| llava4d   | llava-hf/llava-v1.6-mistral-7b-hf | LLaVA-4D pas encore dispo |
| sa2va     | ByteDance/Sa2VA-4B        | trust_remote_code       |
