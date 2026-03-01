#!/usr/bin/env python3
"""
Generate figures for Confidence MAS v2 run: accuracy over steps, score evolution,
assignment changes, and final score map. Uses parsed log data from test_confidence_mas_v2.

Output: docs/fig_confidence_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parsed from CV-Bench 49-sample run (2026-03-01)
# Step N: acc after N samples, category, assignment
STEPS_DATA = [
    (2, 50.0, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (3, 66.7, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (4, 75.0, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "llava4d")]),
    (5, 80.0, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "llava4d")]),
    (6, 83.3, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (7, 85.7, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (8, 75.0, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (9, 77.8, "spatial_relation", [("direct_visual_heuristic", "llava4d"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "spatial_reasoner")]),
    (10, 80.0, "counting", [("direct_visual_heuristic", "llava4d"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "qwen3_4b")]),
    (11, 81.8, "spatial_relation", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (12, 83.3, "counting", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (13, 84.6, "spatial_relation", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (14, 78.6, "counting", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (15, 80.0, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (16, 81.2, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (17, 82.4, "spatial_relation", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (18, 83.3, "spatial_relation", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "qwen3_4b"), ("scene_graph_construction", "llava4d")]),
    (19, 84.2, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (20, 80.0, "spatial_relation", [("direct_visual_heuristic", "spatial_reasoner"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "qwen3_4b")]),
    (21, 81.0, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "llava4d")]),
    (22, 81.8, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (23, 82.6, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (24, 83.3, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "llava4d")]),
    (25, 84.0, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "spatial_reasoner"), ("scene_graph_construction", "llava4d")]),
    (26, 84.6, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (27, 85.2, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (28, 85.7, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (29, 86.2, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (30, 86.7, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (31, 87.1, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (32, 87.5, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (33, 87.9, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (34, 88.2, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (35, 88.6, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (36, 88.9, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (37, 89.2, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (38, 86.8, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (39, 87.2, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (40, 87.5, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (41, 87.8, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (42, 88.1, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (43, 88.4, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (44, 88.6, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (45, 88.9, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (46, 89.1, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (47, 87.2, "counting", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (48, 87.5, "spatial_relation", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
    (49, 87.8, "distance_depth", [("direct_visual_heuristic", "qwen3_4b"), ("explicit_3d_representation", "llava4d"), ("scene_graph_construction", "spatial_reasoner")]),
]

# Final score map (from step 48)
FINAL_SCORE_MAP = {
    "spatial_relation": {
        "direct_visual_heuristic": {"qwen3_4b": 1.0, "llava4d": 0.0, "spatial_reasoner": 0.0},
        "explicit_3d_representation": {"qwen3_4b": 0.0, "llava4d": 1.0, "spatial_reasoner": 0.0},
        "scene_graph_construction": {"qwen3_4b": 0.0, "llava4d": 1.0, "spatial_reasoner": 1.0},
    },
    "distance_depth": {
        "direct_visual_heuristic": {"qwen3_4b": 1.0, "llava4d": 0.5, "spatial_reasoner": 0.5},
        "explicit_3d_representation": {"qwen3_4b": 0.5, "llava4d": 0.0, "spatial_reasoner": 0.0},
        "scene_graph_construction": {"qwen3_4b": 0.5, "llava4d": 0.0, "spatial_reasoner": 1.0},
    },
    "size": {
        "direct_visual_heuristic": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
        "explicit_3d_representation": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
        "scene_graph_construction": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
    },
    "orientation": {
        "direct_visual_heuristic": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
        "explicit_3d_representation": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
        "scene_graph_construction": {"qwen3_4b": 0.5, "llava4d": 0.5, "spatial_reasoner": 0.5},
    },
    "counting": {
        "direct_visual_heuristic": {"qwen3_4b": 0.0, "llava4d": 0.0, "spatial_reasoner": 0.0},
        "explicit_3d_representation": {"qwen3_4b": 0.0, "llava4d": 0.0, "spatial_reasoner": 0.0},
        "scene_graph_construction": {"qwen3_4b": 1.0, "llava4d": 0.0, "spatial_reasoner": 0.0},
    },
}

# Category-wise final accuracy
CAT_ACC = {"counting": 81.8, "distance_depth": 100.0, "spatial_relation": 81.0}


def fig1_accuracy_over_steps():
    """Line chart: Accuracy vs step (cumulative)."""
    steps = [s[0] for s in STEPS_DATA]
    accs = [s[1] for s in STEPS_DATA]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, accs, "o-", color="#2ecc71", linewidth=2, markersize=4)
    ax.fill_between(steps, accs, alpha=0.2, color="#2ecc71")
    ax.set_xlabel("Step (samples processed)")
    ax.set_ylabel("Cumulative Accuracy (%)")
    ax.set_title("Confidence MAS v2: Accuracy over Steps (CV-Bench, 49 samples)")
    ax.set_ylim(0, 100)
    ax.axhline(87.8, color="#e74c3c", linestyle="--", alpha=0.7, label="Final: 87.8%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_accuracy_steps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_accuracy_steps.png'}")


def fig2_category_distribution():
    """Bar chart: Category distribution at each step."""
    cats = [s[2] for s in STEPS_DATA]
    from collections import Counter
    cnt = Counter(cats)
    labels = list(cnt.keys())
    vals = [cnt[c] for c in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, vals, color=colors[:len(labels)])
    ax.set_ylabel("Sample count")
    ax.set_title("Category Distribution in 49 Samples")
    for b in bars:
        ax.annotate(f"{b.get_height()}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_category_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_category_dist.png'}")


def fig3_assignment_changes():
    """Show which LLM was assigned to each role over steps (as stacked area or timeline)."""
    roles = ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]
    llms = ["qwen3_4b", "llava4d", "spatial_reasoner"]
    llm_map = {"qwen3_4b": 0, "llava4d": 1, "spatial_reasoner": 2}

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]

    for ri, role in enumerate(roles):
        ax = axes[ri]
        step_assign = []
        for s in STEPS_DATA:
            for r, llm in s[3]:
                if r == role:
                    step_assign.append(llm_map.get(llm, -1))
                    break
        steps = [s[0] for s in STEPS_DATA]
        # Plot as discrete steps
        for i, (st, a) in enumerate(zip(steps, step_assign)):
            c = colors[a] if a >= 0 else "gray"
            ax.barh(0, 1, left=st, height=0.5, color=c, alpha=0.8)
        ax.set_ylabel(role.replace("_", "\n")[:20])
        ax.set_xlim(0, 50)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

    axes[0].set_title("LLM Assignment Over Steps (by role)")
    axes[2].set_xlabel("Step")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=llm) for i, llm in enumerate(llms)]
    fig.legend(handles=legend_elements, loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_assignments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_assignments.png'}")


def fig3b_assignment_changes_simple():
    """Simpler: line chart of role assignment changes over time."""
    roles = ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]
    llms = ["qwen3_4b", "llava4d", "spatial_reasoner"]
    llm_to_num = {llm: i for i, llm in enumerate(llms)}

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for ri, role in enumerate(roles):
        ax = axes[ri]
        step_assign = []
        for s in STEPS_DATA:
            for r, llm in s[3]:
                if r == role:
                    step_assign.append(llm_to_num.get(llm, -1))
                    break
        steps = [s[0] for s in STEPS_DATA]
        ax.step(steps, step_assign, where="post", color=colors[ri], linewidth=2)
        short = role.replace("direct_visual_heuristic", "direct").replace("explicit_3d_representation", "explicit_3d").replace("scene_graph_construction", "scene_graph")
        ax.set_ylabel(short[:12])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(llms, fontsize=8)
        ax.set_ylim(-0.3, 2.3)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("LLM Assignment Over Steps (by role)")
    axes[2].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_assignments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_assignments.png'}")


def fig4_final_score_heatmap():
    """Heatmap: Final score map (category × role, value = best LLM score)."""
    cats = list(FINAL_SCORE_MAP.keys())
    roles = ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]
    llms = ["qwen3_4b", "llava4d", "spatial_reasoner"]

    # Build matrix: rows = (cat, role), cols = llms
    rows = []
    row_labels = []
    for cat in cats:
        for role in roles:
            row_labels.append(f"{cat[:8]}\n{role.split('_')[0]}")
            row = [FINAL_SCORE_MAP[cat][role].get(llm, 0) for llm in llms]
            rows.append(row)

    mat = np.array(rows)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(llms, fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=7)

    for i in range(len(rows)):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Confidence score")
    ax.set_title("Final Confidence Score Map (category × role × LLM)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_score_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_score_heatmap.png'}")


def fig5_category_accuracy():
    """Bar chart: Per-category final accuracy."""
    labels = list(CAT_ACC.keys())
    vals = list(CAT_ACC.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, vals, color=colors[:len(labels)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Category Accuracy (CV-Bench, 49 samples)")
    ax.set_ylim(0, 100)
    ax.axhline(87.8, color="gray", linestyle="--", alpha=0.5, label="Overall: 87.8%")
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_category_acc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_category_acc.png'}")


def fig6_accuracy_vs_rolling():
    """Accuracy + rolling average."""
    steps = [s[0] for s in STEPS_DATA]
    accs = [s[1] for s in STEPS_DATA]
    window = 5
    rolling = np.convolve(accs, np.ones(window)/window, mode="valid")
    rolling_steps = steps[window-1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, accs, "o-", color="#2ecc71", linewidth=1.5, markersize=3, alpha=0.7, label="Cumulative")
    ax.plot(rolling_steps, rolling, "-", color="#e74c3c", linewidth=2, label=f"Rolling avg (window={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: Cumulative vs Rolling Average")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_rolling_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'fig_confidence_rolling_accuracy.png'}")


def main():
    fig1_accuracy_over_steps()
    fig2_category_distribution()
    fig3b_assignment_changes_simple()
    fig4_final_score_heatmap()
    fig5_category_accuracy()
    fig6_accuracy_vs_rolling()
    print("Done. Figures saved to docs/fig_confidence_*.png")


if __name__ == "__main__":
    main()
