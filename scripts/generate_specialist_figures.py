#!/usr/bin/env python3
"""
Generate figures for Specialist 5 Models × 3 Roles × 2 Benchmarks results.
Output: docs/fig_specialist_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data: Model -> Role -> (CV%, 3D%)
# direct_visual, explicit_3d, scene_graph
DATA = {
    "Qwen3-4B": [(88.9, 50.0), (77.8, 50.0), (88.9, 40.0)],
    "Sa2VA": [(70.0, 60.0), (70.0, 50.0), (70.0, 60.0)],
    "SpatialReasoner": [(70.0, 60.0), (80.0, 50.0), (60.0, 60.0)],
    "LLaVA4D": [(60.0, 50.0), (60.0, 30.0), (50.0, 40.0)],
    "SpatialRGPT": [(50.0, 60.0), (60.0, 40.0), (60.0, 50.0)],
}

MODELS = list(DATA.keys())
ROLES = ["direct_visual", "explicit_3d", "scene_graph"]
ROLE_LABELS = ["Direct Visual", "Explicit 3D", "Scene Graph"]


def fig1_bar_model_by_benchmark():
    """Bar chart: Each model's CV-Bench vs 3DSRBench (averaged across roles)."""
    cv_avg = [np.mean([DATA[m][r][0] for r in range(3)]) for m in MODELS]
    d3_avg = [np.mean([DATA[m][r][1] for r in range(3)]) for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, cv_avg, w, label="CV-Bench", color="#2ecc71")
    bars2 = ax.bar(x + w/2, d3_avg, w, label="3DSRBench", color="#3498db")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Specialist Models: CV-Bench vs 3DSRBench (avg across 3 roles)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(70, color="gray", linestyle="--", alpha=0.5)

    for b in bars1:
        ax.annotate(f"{b.get_height():.0f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    for b in bars2:
        ax.annotate(f"{b.get_height():.0f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_specialist_cv_vs_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_specialist_cv_vs_3d.png'}")


def fig2_heatmap():
    """Heatmap: Model × (Role × Benchmark)."""
    # Rows: models, Cols: direct_CV, direct_3D, explicit_CV, explicit_3D, scene_CV, scene_3D
    cols = []
    for r in range(3):
        cols.append(f"{ROLE_LABELS[r][:4]}\nCV")
        cols.append(f"{ROLE_LABELS[r][:4]}\n3D")

    mat = np.zeros((len(MODELS), 6))
    for i, m in enumerate(MODELS):
        for r in range(3):
            mat[i, r*2] = DATA[m][r][0]
            mat[i, r*2+1] = DATA[m][r][1]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title("Accuracy Heatmap: Model × (Role × Benchmark)")

    for i in range(len(MODELS)):
        for j in range(6):
            ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center", fontsize=9, color="black")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_specialist_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_specialist_heatmap.png'}")


def fig3_model_strengths():
    """Bar chart: Best model per (role, benchmark) combination."""
    wins = {m: 0 for m in MODELS}
    for r in range(3):
        for b in range(2):
            vals = [(DATA[m][r][b], m) for m in MODELS]
            best = max(vals, key=lambda x: x[0])
            wins[best[1]] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    bars = ax.bar(MODELS, [wins[m] for m in MODELS], color=colors[:len(MODELS)])
    ax.set_ylabel("# Best (Role × Benchmark)")
    ax.set_title("Model Strength: How many (role, benchmark) combos each model wins")
    ax.set_ylim(0, 6)
    for b in bars:
        ax.annotate(f"{int(b.get_height())}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_specialist_strengths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_specialist_strengths.png'}")


def fig4_role_comparison():
    """Grouped bar: For each role, compare models on CV vs 3D."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for r, (ax, role) in enumerate(zip(axes, ROLES)):
        cv_vals = [DATA[m][r][0] for m in MODELS]
        d3_vals = [DATA[m][r][1] for m in MODELS]
        x = np.arange(len(MODELS))
        w = 0.35
        ax.bar(x - w/2, cv_vals, w, label="CV-Bench", color="#2ecc71")
        ax.bar(x + w/2, d3_vals, w, label="3DSRBench", color="#3498db")
        ax.set_title(ROLE_LABELS[r])
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        if r == 0:
            ax.legend()
            ax.set_ylabel("Accuracy (%)")
    fig.suptitle("Accuracy by Role: CV-Bench vs 3DSRBench", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_specialist_by_role.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_specialist_by_role.png'}")


def fig5_task_tendency():
    """Summary: Which benchmark favors which model tendency."""
    # CV-Bench best: Qwen3 (89%), SpatialReasoner explicit_3d (80%)
    # 3DSRBench best: SpatialRGPT direct (60%), Sa2VA (60%)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    text = """
    CV-Bench tendency:
    • Qwen3-4B dominates (77–89% across roles) — strong on Count, Distance
    • SpatialReasoner + explicit_3d excels (80%)
    • LLaVA4D weakest (50–60%)

    3DSRBench tendency:
    • SpatialRGPT direct_visual best (60%) — image-only works well for 3D
    • Sa2VA, SpatialReasoner stable (50–60%)
    • LLaVA4D explicit_3d worst (30%) — 3D tool hurts

    Task strengths:
    • Count: Qwen3 100%, LLaVA4D 0%
    • Relation: Sa2VA 80–100%, Qwen3 67%
    • location_above: SpatialRGPT/SpatialReasoner 100%
    • multi_object_*: All models struggle (0–50%)
    """
    ax.text(0.5, 0.5, text.strip(), transform=ax.transAxes,
            fontsize=11, verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            family="monospace")
    ax.set_title("Task & Benchmark Tendencies", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_specialist_tendencies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_specialist_tendencies.png'}")


def main():
    fig1_bar_model_by_benchmark()
    fig2_heatmap()
    fig3_model_strengths()
    fig4_role_comparison()
    fig5_task_tendency()
    print("Done. Figures saved to docs/")


if __name__ == "__main__":
    main()
