#!/usr/bin/env python3
"""
Verify that each SFT model loads and runs inference (no training).

Runs 1 CV-Bench sample per model to confirm:
- Model loads successfully
- Inference completes without error

Usage:
  python scripts/sft_cvbench/00_verify_models.py
  python scripts/sft_cvbench/00_verify_models.py --model qwen3_4b
  python scripts/sft_cvbench/00_verify_models.py --model sa2va
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml


def load_config():
    config_path = Path(__file__).parent / "config_sft.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_qwen3_4b(model_id: str) -> Tuple[bool, str]:
    """Load Qwen3-VL and run 1 inference."""
    try:
        from src.models.qwen3 import Qwen3Runner
        from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_image
    except ImportError as e:
        return False, f"Import: {e}"

    try:
        runner = Qwen3Runner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        ds = load_benchmark("cvbench", max_samples=1, seed=42)
        ex = ds[0]
        img = get_benchmark_image(ex, "cvbench")
        prompt = get_benchmark_prompt(ex, "cvbench")
        if img is None:
            return False, "No image in sample"
        out = runner.generate(img, prompt, max_new_tokens=64)
        if not isinstance(out, str):
            return False, f"Expected str, got {type(out)}"
        return True, f"OK (len={len(out)})"
    except Exception as e:
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_llava4d(model_id: str) -> Tuple[bool, str]:
    """Load LLaVA4D (LLaVA-NeXT) and run 1 inference."""
    try:
        from src.models.llava import LLaVARunner
        from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_image
    except ImportError as e:
        return False, f"Import: {e}"

    try:
        runner = LLaVARunner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        ds = load_benchmark("cvbench", max_samples=1, seed=42)
        ex = ds[0]
        img = get_benchmark_image(ex, "cvbench")
        prompt = get_benchmark_prompt(ex, "cvbench")
        if img is None:
            return False, "No image in sample"
        out = runner.generate(img, prompt, max_new_tokens=64)
        if not isinstance(out, str):
            return False, f"Expected str, got {type(out)}"
        return True, f"OK (len={len(out)})"
    except Exception as e:
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_sa2va(model_id: str) -> Tuple[bool, str]:
    """Load Sa2VA and run 1 inference."""
    try:
        from src.models.sa2va import Sa2VARunner
        from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_image
    except ImportError as e:
        return False, f"Import: {e}"

    try:
        runner = Sa2VARunner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        ds = load_benchmark("cvbench", max_samples=1, seed=42)
        ex = ds[0]
        img = get_benchmark_image(ex, "cvbench")
        prompt = get_benchmark_prompt(ex, "cvbench")
        if img is None:
            return False, "No image in sample"
        out = runner.generate(img, prompt, max_new_tokens=64)
        if not isinstance(out, str):
            return False, f"Expected str, got {type(out)}"
        return True, f"OK (len={len(out)})"
    except Exception as e:
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_spatialrgpt(model_id: str) -> Tuple[bool, str]:
    """Load SpatialRGPT (requires SPATIALRGPT_PATH)."""
    import os
    if not os.environ.get("SPATIALRGPT_PATH") or not Path(os.environ["SPATIALRGPT_PATH"]).is_dir():
        return False, "SPATIALRGPT_PATH not set or invalid. Clone repo and set: export SPATIALRGPT_PATH=/path/to/SpatialRGPT"

    try:
        from src2.models.spatial_rgpt import SpatialRGPTRunner
        from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_image
    except ImportError as e:
        return False, f"Import: {e}"

    try:
        runner = SpatialRGPTRunner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        ds = load_benchmark("cvbench", max_samples=1, seed=42)
        ex = ds[0]
        img = get_benchmark_image(ex, "cvbench")
        prompt = get_benchmark_prompt(ex, "cvbench")
        if img is None:
            return False, "No image in sample"
        out = runner.generate(img, prompt, max_new_tokens=64)
        if not isinstance(out, str):
            return False, f"Expected str, got {type(out)}"
        return True, f"OK (len={len(out)})"
    except Exception as e:
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_spatialreasoner(model_id: str) -> Tuple[bool, str]:
    """Load SpatialReasoner and run 1 inference."""
    try:
        from src2.models.spatial_reasoner import SpatialReasonerRunner
        from src.benchmarks import load_benchmark, get_benchmark_prompt, get_benchmark_image
    except ImportError as e:
        return False, f"Import: {e}"

    try:
        runner = SpatialReasonerRunner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        ds = load_benchmark("cvbench", max_samples=1, seed=42)
        ex = ds[0]
        img = get_benchmark_image(ex, "cvbench")
        prompt = get_benchmark_prompt(ex, "cvbench")
        if img is None:
            return False, "No image in sample"
        out = runner.generate(img, prompt, max_new_tokens=64)
        if not isinstance(out, str):
            return False, f"Expected str, got {type(out)}"
        return True, f"OK (len={len(out)})"
    except Exception as e:
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


VERIFIERS = {
    "qwen3_4b": (verify_qwen3_4b, "Qwen/Qwen3-VL-4B-Instruct"),
    "llava4d": (verify_llava4d, "llava-hf/llava-v1.6-mistral-7b-hf"),
    "sa2va": (verify_sa2va, "ByteDance/Sa2VA-4B"),
    "spatialrgpt": (verify_spatialrgpt, "a8cheng/SpatialRGPT-VILA1.5-8B"),
    "spatialreasoner": (verify_spatialreasoner, "ccvl/SpatialReasoner"),
}


def main():
    parser = argparse.ArgumentParser(description="Verify SFT models load and run inference")
    parser.add_argument("--model", type=str, default=None, choices=list(VERIFIERS) + [None],
                        help="Single model to verify; default: all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    models_cfg = {}
    root_cfg_path = ROOT / "config.yaml"
    if root_cfg_path.exists():
        with open(root_cfg_path, "r") as f:
            root_cfg = yaml.safe_load(f)
            models_cfg = root_cfg.get("models", {})

    models = [args.model] if args.model else list(VERIFIERS)
    print("=" * 70)
    print("SFT Model Verification (load + 1 inference, no training)")
    print("=" * 70)
    print(f"Models: {models}")
    print()

    all_ok = True
    for name in models:
        verifier, default_id = VERIFIERS[name]
        model_id = models_cfg.get(name, {}).get("model_id", default_id)
        print(f"[{name}] {model_id} ... ", end="", flush=True)
        ok, msg = verifier(model_id)
        if ok:
            print(f"PASS — {msg}")
        else:
            print(f"FAIL — {msg}")
            all_ok = False

    print()
    print("=" * 70)
    if all_ok:
        print("All models verified OK.")
    else:
        print("Some models failed. Install missing deps (e.g. accelerate, transformers>=4.57).")
        print("  spatialrgpt: export SPATIALRGPT_PATH=/path/to/SpatialRGPT")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
