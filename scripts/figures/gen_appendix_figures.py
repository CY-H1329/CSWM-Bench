#!/usr/bin/env python3
"""
Generate all figures for the Appendix GitHub issue.
Toss-style color palette. Run from project root: python scripts/figures/gen_appendix_figures.py
"""
import json
import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Toss brand colors (clean, minimal)
TOSS = {
    "blue": "#0064FF",       # Toss Blue (primary)
    "blue_light": "#3182F6",
    "blue_bg": "#E8F3FF",    # Light blue background
    "mint": "#00C48C",       # Success / positive
    "mint_bg": "#E6FAF5",
    "gray": "#202632",       # Toss Gray (text)
    "gray_light": "#F2F4F6", # Card background
    "gray_border": "#E8EBEF",
    "white": "#FFFFFF",
}

# Agent profile display names
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
    """Generate model x category performance heatmap (original RdYlGn)."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skipping heatmap (missing deps): {e}")
        return None

    data = load_agent_profiles()
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
    fig.patch.set_facecolor(TOSS["white"])
    ax.set_facecolor(TOSS["white"])

    # Original RdYlGn colormap (red=low, yellow=mid, green=high)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", color=TOSS["gray"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color=TOSS["gray"])
    ax.set_title("Head-Agent Model Profiles: Per-Category Performance (Unified)", color=TOSS["gray"], fontsize=11)

    for i in range(len(models)):
        for j in range(len(cats)):
            val = arr[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Performance (0–1)", color=TOSS["gray"])
    cbar.ax.yaxis.set_tick_params(color=TOSS["gray"])
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_model_profiles_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=TOSS["white"])
    plt.close()
    print(f"Saved {out}")
    return out


def gen_role_tool_matrix():
    """Generate Role x Tool matrix (Toss colors)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skipping role-tool matrix: {e}")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(TOSS["white"])
    ax.set_facecolor(TOSS["white"])
    ax.axis("off")

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

    # Toss styling
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor(TOSS["white"])
        cell.set_edgecolor(TOSS["gray_border"])
        cell.set_text_props(color=TOSS["gray"])
        if i == 0:
            cell.set_facecolor(TOSS["blue_bg"])
            cell.set_text_props(weight="bold", color=TOSS["blue"])
        elif "✓" in str(cell.get_text().get_text()):
            cell.set_facecolor(TOSS["mint_bg"])
            cell.set_text_props(color=TOSS["mint"])

    ax.set_title("Specialist Role × Tool Assignment", color=TOSS["gray"], fontsize=11)
    out = OUTPUT_DIR / "fig_role_tool_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.5, facecolor=TOSS["white"])
    plt.close()
    print(f"Saved {out}")
    return out


def gen_final_reasoning_flow():
    """Generate Final Reasoning 5-step protocol. Arrows: Step 1 → 2 → 3 → 4 → 5 (downward)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        print(f"Skipping flow: {e}")
        return None

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.patch.set_facecolor(TOSS["white"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_facecolor(TOSS["white"])
    ax.axis("off")

    steps = [
        (5, 10, "Step 1: Understand the question"),
        (5, 8.5, "Step 2: Read each agent's reasoning"),
        (5, 7, "Step 3: Compare and synthesize"),
        (5, 5.5, "Step 4: Draw conclusion"),
        (5, 4, "Step 5: Output (Answer + Reason)"),
    ]

    for i, (x, y, text) in enumerate(steps):
        # Toss colors: blue for steps 1-4, mint for final step
        facecolor = TOSS["blue_bg"] if i < 4 else TOSS["mint_bg"]
        edgecolor = TOSS["blue"] if i < 4 else TOSS["mint"]
        box = FancyBboxPatch((1, y - 0.4), 8, 0.8, boxstyle="round,pad=0.05",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, y, text, ha="center", va="center", fontsize=9, color=TOSS["gray"])

        # Arrow: FROM current step (bottom) TO next step (top). Direction: DOWN.
        if i < 4:
            y_next = steps[i + 1][1]
            # Start: bottom of current box (y - 0.4), End: top of next box (y_next + 0.4)
            ax.annotate("", xy=(5, y_next + 0.4), xytext=(5, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color=TOSS["blue"], lw=2))

    ax.text(5, 11, "Final Reasoning Protocol", ha="center", fontsize=12, fontweight="bold", color=TOSS["gray"])
    ax.text(5, 0.5, "Input: Question + SharedMemory (3 specialist outputs)", ha="center", fontsize=8, color=TOSS["gray"], style="italic")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_final_reasoning_protocol.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=TOSS["white"])
    plt.close()
    print(f"Saved {out}")
    return out


def gen_mas_pipeline_diagram():
    """Generate MAS pipeline flow. Correct flow: Image+Query → Head → ScoreMap → 3 Specialists → SharedMemory → Final Reasoning → Answer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        print(f"Skipping pipeline: {e}")
        return None

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(TOSS["white"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_facecolor(TOSS["white"])
    ax.axis("off")

    # Vertical positions (top to bottom)
    y_input = 10
    y_head = 8.5
    y_scoremap = 7
    y_specialists = 5
    y_shared = 3
    y_final = 1.5
    y_output = 0.3

    # Input
    ax.add_patch(FancyBboxPatch((3.5, y_input - 0.35), 3, 0.7, boxstyle="round,pad=0.05",
                                facecolor=TOSS["gray_light"], edgecolor=TOSS["gray_border"]))
    ax.text(5, y_input, "Image + Query", ha="center", va="center", fontsize=9, color=TOSS["gray"])

    # Head Agent
    ax.add_patch(FancyBboxPatch((3.5, y_head - 0.35), 3, 0.7, boxstyle="round,pad=0.05",
                                facecolor=TOSS["blue_bg"], edgecolor=TOSS["blue"]))
    ax.text(5, y_head, "Head Agent (category)", ha="center", va="center", fontsize=9, color=TOSS["gray"])

    # ScoreMap
    ax.add_patch(FancyBboxPatch((3.5, y_scoremap - 0.35), 3, 0.7, boxstyle="round,pad=0.05",
                                facecolor=TOSS["blue_bg"], edgecolor=TOSS["blue"]))
    ax.text(5, y_scoremap, "ScoreMap (role × llm)", ha="center", va="center", fontsize=9, color=TOSS["gray"])

    # Three specialists (horizontal)
    specialists = [("Direct Visual", 1.5), ("Explicit 3D", 5), ("Scene Graph", 8.5)]
    for name, x in specialists:
        ax.add_patch(FancyBboxPatch((x - 0.6, y_specialists - 0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                    facecolor=TOSS["blue_bg"], edgecolor=TOSS["blue"]))
        ax.text(x, y_specialists, name, ha="center", va="center", fontsize=8, color=TOSS["gray"])

    # SharedMemory
    ax.add_patch(FancyBboxPatch((3, y_shared - 0.35), 4, 0.7, boxstyle="round,pad=0.05",
                                facecolor=TOSS["mint_bg"], edgecolor=TOSS["mint"]))
    ax.text(5, y_shared, "SharedMemory", ha="center", va="center", fontsize=9, color=TOSS["gray"])

    # Final Reasoning
    ax.add_patch(FancyBboxPatch((3.5, y_final - 0.35), 3, 0.7, boxstyle="round,pad=0.05",
                                facecolor=TOSS["mint_bg"], edgecolor=TOSS["mint"]))
    ax.text(5, y_final, "Final Reasoning", ha="center", va="center", fontsize=9, color=TOSS["gray"])

    # Arrows (top to bottom)
    ax.annotate("", xy=(5, y_head + 0.35), xytext=(5, y_input - 0.35),
                arrowprops=dict(arrowstyle="->", color=TOSS["blue"], lw=2))
    ax.annotate("", xy=(5, y_scoremap + 0.35), xytext=(5, y_head - 0.35),
                arrowprops=dict(arrowstyle="->", color=TOSS["blue"], lw=2))

    # ScoreMap → all 3 specialists (branching)
    for _, x in specialists:
        ax.annotate("", xy=(x, y_specialists + 0.4), xytext=(5, y_scoremap - 0.35),
                    arrowprops=dict(arrowstyle="->", color=TOSS["blue"], lw=2))

    # All 3 specialists → SharedMemory (converging)
    for _, x in specialists:
        ax.annotate("", xy=(5, y_shared + 0.35), xytext=(x, y_specialists - 0.4),
                    arrowprops=dict(arrowstyle="->", color=TOSS["mint"], lw=2))

    ax.annotate("", xy=(5, y_final + 0.35), xytext=(5, y_shared - 0.35),
                arrowprops=dict(arrowstyle="->", color=TOSS["mint"], lw=2))
    ax.annotate("", xy=(5, y_output), xytext=(5, y_final - 0.35),
                arrowprops=dict(arrowstyle="->", color=TOSS["mint"], lw=2))

    ax.text(5, 10.7, "Input", ha="center", fontsize=8, color=TOSS["gray"])
    ax.text(5, -0.1, "Final Answer", ha="center", fontsize=8, color=TOSS["gray"])
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_mas_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=TOSS["white"])
    plt.close()
    print(f"Saved {out}")
    return out


def main():
    print("Generating appendix figures (Toss palette)...")
    gen_model_profiles_heatmap()
    gen_role_tool_matrix()
    gen_final_reasoning_flow()
    gen_mas_pipeline_diagram()
    print(f"Done. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
