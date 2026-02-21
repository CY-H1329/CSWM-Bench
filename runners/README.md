# Spatial MAS — Agent Runners

Runners pour chaque agent du pipeline MAS (Head → Perception → Reasoning).

## Agents

| Agent | Rôle MAS | Modèle | HuggingFace |
|-------|----------|--------|-------------|
| **Qwen3-4B** | Head + Perception (Direct 2D) | Qwen/Qwen3-VL-4B-Instruct | [huggingface.co/Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| **Sa2VA** | Perception (3D/depth) | ByteDance/Sa2VA-4B | [huggingface.co/ByteDance/Sa2VA-4B](https://huggingface.co/ByteDance/Sa2VA-4B) |
| **LLaVA-4D** | Perception (Direct/relation) | llava-hf/llava-v1.6-mistral-7b-hf | [huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) |
| **SpatialRGPT** | Perception (SceneGraph) | a8cheng/SpatialRGPT-VILA1.5-8B | [huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B](https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B) |
| **SpatialReasoner** | Perception (3D) | ccvl/SpatialReasoner | [huggingface.co/ccvl/SpatialReasoner](https://huggingface.co/ccvl/SpatialReasoner) |
| **DeepSeek-R1** | Reasoning (text-only) | deepseek-ai/DeepSeek-R1 | [huggingface.co/deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |

## Utilisation

```python
# Via le registry (recommandé)
from src.models import get_runner

runner = get_runner("qwen3_4b", device="cuda")
out = runner.generate(image, prompt)
```

Ou importer directement depuis `src.models`:
- `from src.models.qwen3 import Qwen3Runner`
- `from src.models.sa2va import Sa2VARunner`
- etc.

## Prérequis

- `transformers>=4.51` (Qwen3, SpatialReasoner)
- `transformers>=4.45` (LLaVA-4D)
- `SPATIALRGPT_PATH` pour SpatialRGPT (repo officiel)
- GPU recommandé pour les modèles vision
