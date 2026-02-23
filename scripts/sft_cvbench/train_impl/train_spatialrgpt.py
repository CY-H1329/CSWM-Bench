"""
SpatialRGPT SFT training on CV-Bench (stub).
Requires SPATIALRGPT_PATH to be set. Uses official SpatialRGPT repo for loading.
Implement full training via SpatialRGPT's training scripts or adapt this stub.
"""
import os
import sys
from pathlib import Path


def train_spatialrgpt(
    train_indices: list,
    output_dir: str,
    model_id: str = "a8cheng/SpatialRGPT-VILA1.5-8B",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_spatial_prompt: bool = False,
):
    """Stub for SpatialRGPT SFT. Implement via SpatialRGPT repo or LLaMA-Factory."""
    repo_path = os.environ.get("SPATIALRGPT_PATH")
    if not repo_path or not Path(repo_path).is_dir():
        raise RuntimeError(
            "SPATIALRGPT_PATH not set or invalid. "
            "Clone https://github.com/AnjieCheng/SpatialRGPT and set: "
            "export SPATIALRGPT_PATH=/path/to/SpatialRGPT"
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # TODO: Implement full SFT using SpatialRGPT's load_pretrained_model and training loop
    # For now, save config for manual training
    import json
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": model_id,
        "train_indices": train_indices[:10],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "spatialrgpt_path": repo_path,
    }
    with open(out_path / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"SpatialRGPT stub: config saved to {out_path / 'run_config.json'}")
    print("  Implement full training via SpatialRGPT repo or LLaMA-Factory.")
