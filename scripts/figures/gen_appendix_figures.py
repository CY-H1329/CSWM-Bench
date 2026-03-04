#!/usr/bin/env python3
"""
Generate all figures for the Appendix GitHub issue.
Run from project root: python scripts/figures/gen_appendix_figures.py
"""
import json
import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Agent profile display names (match heatmap style)
MODEL_DISPLAY_NAMES = {
    "claude_sonnet_4_5": "Claude 4.5",
    "gemini_robotics_er": "Gemini-ER",
    "gpt4o": "GPT-4o",
    "llava4d": "LLaVA-4D",
    "qwen3_4b": "Qwen3-4B",
    "sa2va": "Sa2VA",
}

CATEGORIES_ORDER = [
    "depth", "distance", "relation", "existence", "count",
    "instance_location", "orientation", "size", "reach"
]


def load_agent_profiles():
    """Load unified_per_category from all agent profiles."""
    profiles_dir = PROJECT_ROOT / "configs" / "mas" / "agent_profiles"
    data = {}
    for path in sorted(profiles_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        name = d.get("name", path.stem)
        unified = d.get("unified_per_category", {})
        data[name] = unified
    return data


def gen_model_profiles_heatmap():
    """Generate model x category performance heatmap."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        print(f"Skipping heatmap (missing deps): {e}")
        return None

    data = load_agent_profiles()
    models = [MODEL_DISPLAY_NAMES.get(m, m) for m in sorted(data.keys()) if m in MODEL_DISPLAY_NAMES or m in data]
    # Order: Claude, Gemini, GPT-4o, LLaVA, Qwen3, Sa2VA
    order = ["claude_sonnet_4_5", "gemini_robotics_er", "gpt4o", "llava4d", "qwen3_4b", "sa2va"]
    models = [MODEL_DISPLAY_NAMES.get(m, m) for m in order if m in data]
    cats = [c for c in CATEGORIES_ORDER if any(data[m].get(c) is not None for m in data)]

    matrix = []
    for m in order:
        if m not in data:
            continue
        row = [data[m].get(c, 0) for c in cats]
        matrix.append(row)

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Head-Agent Model Profiles: Per-Category Performance (Unified)")

    for i in range(len(models)):
        for j in range(len(cats)):
            val = arr[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Performance (0–1)")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_model_profiles_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


def gen_role_tool_matrix():
    """Generate Role x Tool matrix as image."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        print(f"Skipping role-tool matrix: {e}")
        return None

    roles = ["direct_visual_heuristic", "explicit_3d_representation", "scene_graph_construction"]
    tools = ["None", "Depth / 3D Representation", "Scene Graph"]
    # direct: no tool, explicit_3d: depth, scene_graph: scene graph
    matrix = [
        ["✓", "—", "—"],
        ["—", "✓", "—"],
        ["—", "—", "✓"],
    ]
    labels = [["Yes", "No", "No"], ["No", "Yes", "No"], ["No", "No", "Yes"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    role_short = ["Direct Visual", "Explicit 3D", "Scene Graph"]
    col_labels = ["Tool", "direct_visual", "explicit_3d", "scene_graph"]
    table_data = [
        ["Depth / 3D Repr", "—", "✓", "—"],
        ["Scene Graph", "—", "—", "✓"],
        ["None (pictorial only)", "✓", "—", "—"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Tool", "Direct Visual", "Explicit 3D", "Scene Graph"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    ax.set_title("Specialist Role × Tool Assignment")
    out = OUTPUT_DIR / "fig_role_tool_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.5)
    plt.close()
    print(f"Saved {out}")
    return out


def gen_final_reasoning_flow():
    """Generate Final Reasoning 5-step protocol flowchart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        print(f"Skipping flow: {e}")
        return None

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    steps = [
        (5, 10, "Step 1: Understand the question"),
        (5, 8.5, "Step 2: Read each agent's reasoning"),
        (5, 7, "Step 3: Compare and synthesize"),
        (5, 5.5, "Step 4: Draw conclusion"),
        (5, 4, "Step 5: Output (Answer + Reason)"),
    ]
    y_offsets = [10, 8.5, 7, 5.5, 4]
    for i, (x, y, text) in enumerate(steps):
        box = FancyBboxPatch((1, y - 0.4), 8, 0.8, boxstyle="round,pad=0.05",
                             facecolor="lightblue" if i < 4 else "lightgreen", edgecolor="black")
        ax.add_patch(box)
        ax.text(5, y, text, ha="center", va="center", fontsize=9, wrap=True)
        if i < 4:
            ax.annotate("", xy=(5, y - 0.5), xytext=(5, y - 0.9),
                        arrowprops=dict(arrowstyle="->", color="black"))

    ax.text(5, 11, "Final Reasoning Protocol", ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 0.5, "Input: Question + SharedMemory (3 specialist outputs)", ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_final_reasoning_protocol.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


def gen_mas_pipeline_diagram():
    """Generate MAS pipeline flow diagram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        print(f"Skipping pipeline: {e}")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Head
    ax.add_patch(FancyBboxPatch((3.5, 8), 3, 0.8, boxstyle="round,pad=0.05", facecolor="wheat", edgecolor="black"))
    ax.text(5, 8.4, "Head Agent\n(category)", ha="center", va="center", fontsize=9)

    # ScoreMap
    ax.add_patch(FancyBboxPatch((3.5, 6.5), 3, 0.8, boxstyle="round,pad=0.05", facecolor="lightyellow", edgecolor="black"))
    ax.text(5, 6.9, "ScoreMap\n(role, llm) selection", ha="center", va="center", fontsize=9)

    # Three specialists
    for i, (name, x) in enumerate([("Direct Visual", 1), ("Explicit 3D", 4), ("Scene Graph", 7)]):
        ax.add_patch(FancyBboxPatch((x - 0.4, 4), 1.2, 1, boxstyle="round,pad=0.05", facecolor="lightblue", edgecolor="black"))
        ax.text(x, 4.5, name, ha="center", va="center", fontsize=8)

    # SharedMemory
    ax.add_patch(FancyBboxPatch((3, 2), 4, 0.8, boxstyle="round,pad=0.05", facecolor="lavender", edgecolor="black"))
    ax.text(5, 2.4, "SharedMemory", ha="center", va="center", fontsize=9)

    # Final Reasoning
    ax.add_patch(FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.05", facecolor="lightgreen", edgecolor="black"))
    ax.text(5, 0.9, "Final Reasoning", ha="center", va="center", fontsize=9)

    # Arrows
    ax.annotate("", xy=(5, 7.3), xytext=(5, 8), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5, 5), xytext=(5, 6.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5, 2.8), xytext=(5, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5, 1.3), xytext=(5, 2.8), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(5, 9.5, "Image + Query", ha="center", fontsize=8)
    ax.text(5, -0.2, "Final Answer", ha="center", fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_mas_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


def main():
    print("Generating appendix figures...")
    gen_model_profiles_heatmap()
    gen_role_tool_matrix()
    gen_final_reasoning_flow()
    gen_mas_pipeline_diagram()
    print(f"Done. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
