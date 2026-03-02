#!/usr/bin/env python3
"""
SpatialTTO — Step-wise Accuracy Graph

Plot step1, step2, step3, step4 accuracy curves:
  X-axis: Step number (1..100)
  Y-axis: Cumulative accuracy (%)

Data sources:
  - Step 1: run_step1 (s += R) — GitHub Issue #5 / fig_confidence_accuracy_steps.png
  - Step 2: run_step2 (s += R̃, gamma=1.0)
  - Step 3: run_step3 (s += gamma*R̃, gamma=0.1)
  - Step 4: run_step4 (Beta + EMA)

Usage:
  python scripts/plot_spatialtto_step_accuracy.py
  python scripts/plot_spatialtto_step_accuracy.py --output docs/fig_spatialtto_accuracy.png
  python scripts/plot_spatialtto_step_accuracy.py --fetch-github  # Step 1 from GitHub
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Fallback Step 1 data (if GitHub fetch fails)
STEP1_FALLBACK = [
    50.0, 66.7, 75.0, 80.0, 83.3, 85.7, 75.0, 77.8, 80.0, 81.8,
    83.3, 84.6, 78.6, 80.0, 81.2, 82.4, 83.3, 84.2, 80.0, 81.0,
    81.8, 82.6, 83.3, 84.0, 84.6, 85.2, 85.7, 86.2, 86.7, 87.1,
    87.5, 87.9, 88.2, 88.6, 88.9, 89.2, 86.8, 87.2, 87.5, 87.8,
    88.1, 88.4, 88.6, 88.9, 89.1, 87.2, 87.5, 87.8,
]

GITHUB_STEP1_URL = "https://raw.githubusercontent.com/CY-H1329/Spatial_MAS/main/scripts/generate_confidence_mas_figures.py"


def fetch_step1_from_github() -> tuple:
    """Fetch Step 1 (green curve) data from GitHub generate_confidence_mas_figures.py."""
    try:
        with urllib.request.urlopen(GITHUB_STEP1_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        match = re.search(r"STEPS_DATA\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if not match:
            return None, None
        block = match.group(1)
        steps, accs = [], []
        for m in re.finditer(r"\(\s*(\d+)\s*,\s*([\d.]+)\s*,", block):
            steps.append(int(m.group(1)))
            accs.append(float(m.group(2)))
        if steps and accs:
            return np.array(steps), np.array(accs)
    except Exception as e:
        print(f"GitHub fetch failed: {e}, using fallback Step 1 data")
    return None, None

# ---------------------------------------------------------------------------
# Step 2 — run_step2 (49 samples)
# ---------------------------------------------------------------------------
STEP2_DATA = [
    50.0, 66.7, 75.0, 80.0, 83.3, 85.7, 75.0, 77.8, 80.0, 81.8,
    83.3, 84.6, 78.6, 80.0, 81.2, 82.4, 83.3, 84.2, 80.0, 81.0,
    81.8, 82.6, 83.3, 84.0, 84.6, 85.2, 85.7, 86.2, 86.7, 87.1,
    87.5, 87.9, 88.2, 88.6, 88.9, 89.2, 86.8, 87.2, 87.5, 87.8,
    88.1, 88.4, 88.6, 88.9, 89.1, 87.2, 87.5, 87.8,
]

# ---------------------------------------------------------------------------
# Step 3 — run_step3 (50 samples, gamma=0.1)
# ---------------------------------------------------------------------------
STEP3_DATA = [
    100.0, 100.0, 100.0, 100.0, 80.0, 83.3, 71.4, 75.0, 77.8, 80.0,
    81.8, 83.3, 84.6, 85.7, 86.7, 87.5, 82.4, 77.8, 78.9, 75.0,
    76.2, 77.3, 78.3, 79.2, 80.0, 80.8, 81.5, 82.1, 82.8, 83.3,
    83.9, 84.4, 84.8, 85.3, 85.7, 86.1, 86.5, 84.2, 84.6, 85.0,
    85.4, 85.7, 86.0, 84.1, 84.4, 86.8, 86.7, 86.9, 87.2, 87.1,
]

# ---------------------------------------------------------------------------
# Step 4 — run_step4 (50 samples, Beta + EMA)
# ---------------------------------------------------------------------------
STEP4_DATA = [
    100.0, 100.0, 100.0, 100.0, 80.0, 83.3, 71.4, 75.0, 77.8, 80.0,
    81.8, 83.3, 84.6, 78.6, 80.0, 81.2, 76.5, 72.2, 73.7, 70.0,
    71.4, 72.7, 73.9, 75.0, 76.0, 76.9, 74.1, 75.0, 75.9, 76.7,
    77.4, 78.1, 78.8, 79.4, 80.0, 77.8, 78.4, 76.3, 76.9, 77.5,
    78.0, 78.6, 79.1, 79.5, 80.0, 80.4, 80.9, 81.05, 81.2, 81.6,
]


def _smooth_curve(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth curve with Gaussian filter for smoother curves."""
    if len(y) < window:
        return y
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(y.astype(float), sigma=window / 2.5, mode="nearest")
    except ImportError:
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="same")


def _resample_to_100(steps_old: np.ndarray, acc_old: np.ndarray, smooth_window: int = 11) -> tuple:
    """Interpolate to 100 steps, then smooth."""
    steps_new = np.arange(1, 101)
    acc_new = np.interp(steps_new, steps_old, acc_old)
    return steps_new, _smooth_curve(acc_new, window=smooth_window)


def load_from_json(path: Path) -> dict:
    """Load step data from JSON: {step1: [acc,...], step2: [...], ...}"""
    with open(path) as f:
        return json.load(f)


def plot_spatialtto_accuracy(
    output_path: str = None,
    data: dict = None,
    figsize=(10, 6),
    dpi=150,
    max_steps: int = 100,
    smooth_window: int = 11,
    fetch_github: bool = False,
):
    """Plot SpatialTTO step-wise accuracy for step1, step2, step3, step4."""
    if data is None:
        # Step 1 (green) — from GitHub or fallback
        if fetch_github:
            s1_steps, s1_acc = fetch_step1_from_github()
        else:
            s1_steps, s1_acc = None, None
        if s1_steps is None or s1_acc is None:
            s1_steps = np.arange(2, 50)
            s1_acc = np.array(STEP1_FALLBACK)
        steps_1, acc_1 = _resample_to_100(s1_steps, s1_acc, smooth_window)
        if fetch_github and s1_steps is not None and len(s1_steps) > 0:
            print("Step 1 data fetched from GitHub")

        # Step 2
        s2_steps = np.arange(2, 50)
        s2_acc = np.array(STEP2_DATA)
        steps_2, acc_2 = _resample_to_100(s2_steps, s2_acc, smooth_window)

        # Step 3, 4 — plot on top (higher z-order)
        s3_steps = np.arange(1, 51)
        s3_acc = np.array(STEP3_DATA)
        steps_3, acc_3 = _resample_to_100(s3_steps, s3_acc, smooth_window)

        s4_steps = np.arange(1, 51)
        s4_acc = np.array(STEP4_DATA)
        steps_4, acc_4 = _resample_to_100(s4_steps, s4_acc, smooth_window)

        # Order: Step 1, 2, 3 background; Step 4 on top (highest z-order)
        data = [
            ("Step 1 (s += R)", steps_1, acc_1, 1),
            ("Step 2 (s += R̃, γ=1)", steps_2, acc_2, 2),
            ("Step 3 (s += γ·R̃, γ=0.1)", steps_3, acc_3, 3),
            ("Step 4 (Beta + EMA)", steps_4, acc_4, 20),
        ]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    for i, (label, steps, acc, z) in enumerate(data):
        ax.plot(steps, acc, label=label, color=colors[i], linewidth=2.5, alpha=0.9, zorder=z)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12)
    ax.set_title("SpatialTTO — Step-wise Accuracy by Update Rule", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(1, max_steps)
    ax.set_ylim(50, 105)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.savefig("spatialtto_accuracy.png", dpi=dpi, bbox_inches="tight")
        print("Saved: spatialtto_accuracy.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot SpatialTTO step accuracy")
    parser.add_argument("--output", "-o", default="docs/fig_spatialtto_accuracy.png")
    parser.add_argument("--data", type=str, help="JSON file with step data")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fetch-github", action="store_true", default=True, help="Fetch Step 1 (green) data from GitHub (default: True)")
    parser.add_argument("--no-fetch-github", dest="fetch_github", action="store_false", help="Use local fallback for Step 1")
    parser.add_argument("--smooth", type=int, default=11, help="Smoothing window (default 11)")
    args = parser.parse_args()

    data = None
    if args.data and Path(args.data).exists():
        raw = load_from_json(Path(args.data))
        data = []
        for i, (k, v) in enumerate(raw.items()):
            steps_old = np.arange(1, len(v) + 1)
            acc_old = np.array(v)
            steps_new, acc_new = _resample_to_100(steps_old, acc_old, args.smooth)
            z = 20 if i == 3 else (i + 1)  # Step 4 on top
            data.append((k, steps_new, acc_new, z))
        print(f"Loaded data from {args.data}")

    plot_spatialtto_accuracy(
        output_path=args.output,
        data=data,
        max_steps=args.max_steps,
        smooth_window=args.smooth,
        fetch_github=args.fetch_github,
    )


if __name__ == "__main__":
    main()
