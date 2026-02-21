#!/usr/bin/env python3
"""
Test agent runners on H100.
- Step 1: Import check
- Step 2: Load one lightweight model (Qwen3-4B or Sa2VA) and run 1 inference
- Step 3: Optional - test more models if --all

Usage:
  cd Spatial_MAS && python eval_h100/test_runners.py
  python eval_h100/test_runners.py --model qwen3_4b
  python eval_h100/test_runners.py --all  # test all available (slow)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    """Step 1: Verify imports."""
    print("[1/3] Testing imports ...")
    from src.models import get_runner, list_agents
    agents = list_agents()
    print(f"  OK: {len(agents)} agents in registry")
    return agents


def test_model(agent_name: str, device: str = "cuda") -> bool:
    """Load model and run 1 inference."""
    from src.models import get_runner
    from PIL import Image
    import torch

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = device if use_cuda else "cpu"

    runner = get_runner(agent_name, device=dev)
    if runner is None:
        print(f"  SKIP {agent_name}: runner not available")
        return False

    # Text-only (e.g. DeepSeek-R1)
    if agent_name == "deepseek_r1":
        prompt = "What is 2+2? Answer with one number."
        try:
            out = runner.generate(prompt, max_new_tokens=16, temperature=0)
            print(f"  OK {agent_name}: output='{str(out)[:60]}...'")
            return True
        except Exception as e:
            print(f"  FAIL {agent_name}: {e}")
            return False

    # Vision models
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    prompt = "What color is this image? Answer in one word."

    try:
        out = runner.generate(img, prompt, max_new_tokens=32, temperature=0)
        out_str = str(out)[:60]
        print(f"  OK {agent_name}: output='{out_str}...'")
        return True
    except Exception as e:
        print(f"  FAIL {agent_name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_4b", help="Agent to test (qwen3_4b, sa2va, llava4d, ...)")
    ap.add_argument("--all", action="store_true", help="Test all available models (slow)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--skip-inference", action="store_true", help="Only test imports, no model load")
    args = ap.parse_args()

    print("=" * 50)
    print("eval_h100: Runner tests")
    print("=" * 50)

    agents = test_imports()
    if args.skip_inference:
        print("[2/3] Skipping inference (--skip-inference)")
        print("[3/3] Done.")
        return 0

    if args.all:
        print("[2/3] Testing all available models (local only, no API) ...")
        import os
        to_test = ["qwen3_4b", "sa2va", "llava4d", "spatialreasoner", "deepseek_r1"]
        if os.environ.get("SPATIALRGPT_PATH"):
            to_test.append("spatialrgpt")
        ok = 0
        for name in to_test:
            if get_runner(name) is not None:
                if test_model(name, args.device):
                    ok += 1
            else:
                print(f"  SKIP {name}: not available")
        print(f"[3/3] Done. {ok}/{len(to_test)} models OK.")
    else:
        print(f"[2/3] Testing {args.model} ...")
        ok = test_model(args.model, args.device)
        print(f"[3/3] Done. {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
