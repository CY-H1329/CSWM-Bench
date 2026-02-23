#!/usr/bin/env python3
"""
Generate comprehensive visualizations for the A-approach:
16 fine-grained classification → 5 unified categories (post-hoc mapping).

Produces two images:
  1. docs/fig1_architecture_mapping.png  — taxonomy & mapping diagram
  2. docs/fig2_benchmark_results.png     — accuracy charts for both benchmarks
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================================
# Color palette
# =====================================================================
C_SPATIAL   = "#4C72B0"
C_DISTANCE  = "#55A868"
C_ORIENT    = "#8172B3"
C_SIZE      = "#C44E52"
C_COUNT     = "#CCB974"
C_UNKNOWN   = "#999999"

UNIFIED_COLORS = {
    "spatial_relation": C_SPATIAL,
    "distance_depth":   C_DISTANCE,
    "orientation":      C_ORIENT,
    "size":             C_SIZE,
    "counting":         C_COUNT,
    "UNKNOWN":          C_UNKNOWN,
}

# =====================================================================
# Data: 3DSRBench (200 samples)
# =====================================================================
DSRBENCH_FINE = [
    ("location_above",                    74.2, 23, 31, "spatial_relation"),
    ("height_higher",                      0.0,  0, 26, "spatial_relation"),
    ("location_closer_to_camera",        100.0, 30, 30, "distance_depth"),
    ("multi_object_closer_to",            57.1,  8, 14, "distance_depth"),
    ("location_next_to",                 100.0, 14, 14, "spatial_relation"),
    ("orientation_on_the_left",           64.3,  9, 14, "orientation"),
    ("orientation_in_front_of",           88.9,  8,  9, "orientation"),
    ("orientation_viewpoint",             50.0,  5, 10, "orientation"),
    ("multi_object_facing",                0.0,  0,  8, "orientation"),
    ("multi_object_same_direction",       75.0,  9, 12, "orientation"),
    ("multi_object_viewpoint_towards_obj", 0.0,  0, 16, "orientation"),
    ("multi_object_parallel",             87.5, 14, 16, "orientation"),
]

DSRBENCH_UNIFIED = {
    "spatial_relation": (88.7, 63, 71),
    "distance_depth":   (100.0, 44, 44),
    "orientation":      (100.0, 85, 85),
}
DSRBENCH_FINE_OVERALL = 60.0
DSRBENCH_UNIFIED_OVERALL = 96.0

# =====================================================================
# Data: CV-Bench (400 samples)
# =====================================================================
CVBENCH_FINE = [
    ("Count",     100.0, 136, 136, "counting"),
    ("Relation",    0.0,   0,  92, "spatial_relation"),
    ("Depth",       0.0,   0,  77, "distance_depth"),
    ("Distance",   23.2,  22,  95, "distance_depth"),
]

CVBENCH_UNIFIED = {
    "spatial_relation": (75.0, 69, 92),
    "distance_depth":   (97.7, 168, 172),
    "counting":         (100.0, 136, 136),
}
CVBENCH_FINE_OVERALL = 39.5
CVBENCH_UNIFIED_OVERALL = 93.2


# =====================================================================
# FIGURE 1: Architecture & Mapping Diagram
# =====================================================================
def draw_figure1():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        "A-Approach: 16 Fine-Grained → 5 Unified Categories\n"
        "Head Agent classifies into concrete categories; code maps to neuroscience-backed taxonomy",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- Left: Mapping diagram ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_title("Category Mapping (FINE_TO_UNIFIED)", fontsize=14, fontweight="bold")

    fine_cats = [
        ("location_above", C_SPATIAL),
        ("height_higher", C_SPATIAL),
        ("location_next_to", C_SPATIAL),
        ("location_closer_to_camera", C_DISTANCE),
        ("multi_object_closer_to", C_DISTANCE),
        ("orientation_on_the_left", C_ORIENT),
        ("orientation_in_front_of", C_ORIENT),
        ("orientation_viewpoint", C_ORIENT),
        ("multi_object_facing", C_ORIENT),
        ("multi_object_same_direction", C_ORIENT),
        ("multi_object_viewpoint_towards", C_ORIENT),
        ("multi_object_parallel", C_ORIENT),
        ("Count", C_COUNT),
        ("Relation", C_SPATIAL),
        ("Depth", C_DISTANCE),
        ("Distance", C_DISTANCE),
    ]

    unified_cats = [
        ("spatial_relation", C_SPATIAL, "WHERE?"),
        ("distance_depth", C_DISTANCE, "HOW FAR?"),
        ("orientation", C_ORIENT, "WHICH WAY?"),
        ("size", C_SIZE, "HOW BIG?"),
        ("counting", C_COUNT, "HOW MANY?"),
    ]

    left_x = 0.5
    right_x = 8.0
    fine_y_start = 18.5
    fine_spacing = 1.1

    for i, (name, color) in enumerate(fine_cats):
        y = fine_y_start - i * fine_spacing
        ax.add_patch(plt.Rectangle((left_x - 0.3, y - 0.35), 3.8, 0.7,
                     facecolor=color, alpha=0.15, edgecolor=color, linewidth=1))
        ax.text(left_x, y, name, fontsize=8, va="center", fontfamily="monospace",
                color=color, fontweight="bold")

    unified_y_positions = {
        "spatial_relation": fine_y_start - 1 * fine_spacing,
        "distance_depth":   fine_y_start - 4 * fine_spacing,
        "orientation":      fine_y_start - 8.5 * fine_spacing,
        "size":             fine_y_start - 13 * fine_spacing,
        "counting":         fine_y_start - 14.5 * fine_spacing,
    }

    for name, color, question in unified_cats:
        y = unified_y_positions[name]
        ax.add_patch(plt.Rectangle((right_x - 0.5, y - 0.6), 2.5, 1.2,
                     facecolor=color, alpha=0.25, edgecolor=color, linewidth=2,
                     zorder=3))
        ax.text(right_x + 0.75, y + 0.15, name, fontsize=9, va="center", ha="center",
                fontweight="bold", color=color, zorder=4)
        ax.text(right_x + 0.75, y - 0.25, question, fontsize=8, va="center", ha="center",
                fontstyle="italic", color="#666666", zorder=4)

    fine_to_unified_map = {
        0: "spatial_relation", 1: "spatial_relation", 2: "spatial_relation",
        3: "distance_depth", 4: "distance_depth",
        5: "orientation", 6: "orientation", 7: "orientation", 8: "orientation",
        9: "orientation", 10: "orientation", 11: "orientation",
        12: "counting", 13: "spatial_relation", 14: "distance_depth", 15: "distance_depth",
    }

    for i, (_, color) in enumerate(fine_cats):
        y_from = fine_y_start - i * fine_spacing
        unified_name = fine_to_unified_map[i]
        y_to = unified_y_positions[unified_name]
        ax.annotate("", xy=(right_x - 0.5, y_to), xytext=(left_x + 3.5, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.4, lw=0.8))

    # --- Right: Neuroscience grounding ---
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Neuroscience Grounding", fontsize=14, fontweight="bold")

    text = """
COGNITIVE NEUROSCIENCE BASIS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  spatial_relation (WHERE?)
    Kosslyn 1987: Categorical spatial processing
    Left parietal cortex encodes discrete positions
    (above/below, next to, between)

  distance_depth (HOW FAR?)
    Kosslyn 1987: Coordinate spatial processing
    Right parietal cortex encodes metric distances
    and depth from viewer

  orientation (WHICH WAY?)
    Levinson 2003: Frames of reference
    Parietal cortex processes viewpoint-dependent
    directions (facing, left/right, parallel)

  size (HOW BIG?)
    Walsh 2003 (ATOM): Magnitude processing
    Intraparietal sulcus (IPS) processes
    relative magnitude comparisons

  counting (HOW MANY?)
    Walsh 2003 (ATOM): Numerosity processing
    IPS encodes quantity via shared magnitude system

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY PRINCIPLE:
  Model classifies into 16 CONCRETE categories
  (what it's good at), then deterministic code
  maps to 5 ABSTRACT categories (neuroscience).

  Fine-grained accuracy: ~60% (semantic overlap)
  Unified accuracy:      ~95% (clean separation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "Let the model do what it's good at.
   Let the code do what code is good at."
"""

    ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
             fontsize=10, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA",
                       edgecolor="#DEE2E6", linewidth=1.5))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / "fig1_architecture_mapping.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# =====================================================================
# FIGURE 2: Benchmark Results
# =====================================================================
def draw_figure2():
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Head Agent Classification Results — A-Approach (16 fine → 5 unified)\n"
        "Model: Qwen3-VL-4B  |  3DSRBench: 200 samples  |  CV-Bench: 400 samples",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- Panel 1: 3DSRBench fine-grained ---
    ax1 = fig.add_subplot(2, 3, 1)
    names = [c[0] for c in DSRBENCH_FINE]
    accs = [c[1] for c in DSRBENCH_FINE]
    totals = [c[3] for c in DSRBENCH_FINE]
    colors = [UNIFIED_COLORS[c[4]] for c in DSRBENCH_FINE]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, accs, color=colors, edgecolor="white", height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=7.5)
    ax1.set_xlim(0, 120)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title(f"3DSRBench Fine-Grained: {DSRBENCH_FINE_OVERALL}%",
                  fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.axvline(x=DSRBENCH_FINE_OVERALL, color="red", linestyle="--", alpha=0.6)

    for i, (acc, n) in enumerate(zip(accs, totals)):
        ax1.text(acc + 1.5, i, f"{acc:.0f}% (n={n})", va="center", fontsize=7)

    # --- Panel 2: 3DSRBench unified ---
    ax2 = fig.add_subplot(2, 3, 2)
    cats_3d = list(DSRBENCH_UNIFIED.keys())
    accs_3d = [DSRBENCH_UNIFIED[c][0] for c in cats_3d]
    totals_3d = [DSRBENCH_UNIFIED[c][2] for c in cats_3d]
    colors_3d = [UNIFIED_COLORS[c] for c in cats_3d]

    y_pos2 = np.arange(len(cats_3d))
    ax2.barh(y_pos2, accs_3d, color=colors_3d, edgecolor="white", height=0.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(cats_3d, fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title(f"3DSRBench Unified: {DSRBENCH_UNIFIED_OVERALL}%",
                  fontsize=12, fontweight="bold", color="#2ECC71")
    ax2.invert_yaxis()
    ax2.axvline(x=DSRBENCH_UNIFIED_OVERALL, color="green", linestyle="--", alpha=0.6)

    for i, (acc, n) in enumerate(zip(accs_3d, totals_3d)):
        ax2.text(acc + 1, i, f"{acc:.1f}% ({n})", va="center", fontsize=10,
                 fontweight="bold")

    # --- Panel 3: Overall comparison ---
    ax3 = fig.add_subplot(2, 3, 3)
    labels = ["3DSR\nFine", "3DSR\nUnified", "CV\nFine", "CV\nUnified"]
    values = [DSRBENCH_FINE_OVERALL, DSRBENCH_UNIFIED_OVERALL,
              CVBENCH_FINE_OVERALL, CVBENCH_UNIFIED_OVERALL]
    bar_colors = ["#E74C3C", "#2ECC71", "#E74C3C", "#2ECC71"]
    bars3 = ax3.bar(labels, values, color=bar_colors, width=0.6, edgecolor="white")
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Fine-Grained vs Unified", fontsize=12, fontweight="bold")

    for bar, val in zip(bars3, values):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val}%",
                 ha="center", fontsize=13, fontweight="bold")

    ax3.annotate(
        f"+{DSRBENCH_UNIFIED_OVERALL - DSRBENCH_FINE_OVERALL:.0f}pp",
        xy=(1, DSRBENCH_UNIFIED_OVERALL), xytext=(0.5, 80),
        fontsize=12, fontweight="bold", color="#2ECC71",
        arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2), ha="center")
    ax3.annotate(
        f"+{CVBENCH_UNIFIED_OVERALL - CVBENCH_FINE_OVERALL:.1f}pp",
        xy=(3, CVBENCH_UNIFIED_OVERALL), xytext=(2.5, 75),
        fontsize=12, fontweight="bold", color="#2ECC71",
        arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2), ha="center")

    # --- Panel 4: CV-Bench fine-grained ---
    ax4 = fig.add_subplot(2, 3, 4)
    cv_names = [c[0] for c in CVBENCH_FINE]
    cv_accs = [c[1] for c in CVBENCH_FINE]
    cv_totals = [c[3] for c in CVBENCH_FINE]
    cv_colors = [UNIFIED_COLORS[c[4]] for c in CVBENCH_FINE]

    y_pos4 = np.arange(len(cv_names))
    ax4.barh(y_pos4, cv_accs, color=cv_colors, edgecolor="white", height=0.5)
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(cv_names, fontsize=11)
    ax4.set_xlim(0, 120)
    ax4.set_xlabel("Accuracy (%)")
    ax4.set_title(f"CV-Bench Fine-Grained: {CVBENCH_FINE_OVERALL}%",
                  fontsize=12, fontweight="bold")
    ax4.invert_yaxis()
    ax4.axvline(x=CVBENCH_FINE_OVERALL, color="red", linestyle="--", alpha=0.6)

    for i, (acc, n) in enumerate(zip(cv_accs, cv_totals)):
        ax4.text(acc + 1.5, i, f"{acc:.0f}% (n={n})", va="center", fontsize=10)

    # --- Panel 5: CV-Bench unified ---
    ax5 = fig.add_subplot(2, 3, 5)
    cats_cv = list(CVBENCH_UNIFIED.keys())
    accs_cv = [CVBENCH_UNIFIED[c][0] for c in cats_cv]
    totals_cv = [CVBENCH_UNIFIED[c][2] for c in cats_cv]
    colors_cv = [UNIFIED_COLORS[c] for c in cats_cv]

    y_pos5 = np.arange(len(cats_cv))
    ax5.barh(y_pos5, accs_cv, color=colors_cv, edgecolor="white", height=0.5)
    ax5.set_yticks(y_pos5)
    ax5.set_yticklabels(cats_cv, fontsize=11, fontweight="bold")
    ax5.set_xlim(0, 115)
    ax5.set_xlabel("Accuracy (%)")
    ax5.set_title(f"CV-Bench Unified: {CVBENCH_UNIFIED_OVERALL}%",
                  fontsize=12, fontweight="bold", color="#2ECC71")
    ax5.invert_yaxis()
    ax5.axvline(x=CVBENCH_UNIFIED_OVERALL, color="green", linestyle="--", alpha=0.6)

    for i, (acc, n) in enumerate(zip(accs_cv, totals_cv)):
        ax5.text(acc + 1, i, f"{acc:.1f}% ({n})", va="center", fontsize=10,
                 fontweight="bold")

    # --- Panel 6: Confusion summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    summary = """CONFUSION ANALYSIS

3DSRBench (200 samples):
  Fine-grained → 60.0% | Unified → 96.0%  (+36pp)

  spatial_relation  88.7% (63/71)
    ▸ 8 samples → UNKNOWN (model answered question)
  distance_depth   100.0% (44/44)  ✓ Perfect
  orientation      100.0% (85/85)  ✓ Perfect

  Key confusion (fine-grained):
    height_higher → location_above (26/26)
    multi_object_facing → other orientation cats
    viewpoint_towards_obj → other orientation cats

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CV-Bench (400 samples):
  Fine-grained → 39.5% | Unified → 93.2%  (+53.7pp)

  spatial_relation  75.0% (69/92)
    ▸ 13 samples → UNKNOWN
    ▸ 6 samples → distance_depth
  distance_depth    97.7% (168/172)  ✓
    ▸ Depth→location_closer_to_camera (remapped)
    ▸ Distance→multi_object_closer_to (remapped)
  counting         100.0% (136/136)  ✓ Perfect

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REMAINING ERRORS: mainly UNKNOWN outputs
  → Model sometimes answers instead of classifying
  → Solvable with output format enforcement"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
             fontsize=8.5, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA",
                       edgecolor="#DEE2E6"))

    # Legend
    patches = [mpatches.Patch(color=UNIFIED_COLORS[c], label=c)
               for c in ["spatial_relation", "distance_depth", "orientation",
                          "size", "counting"]]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    path = OUT_DIR / "fig2_benchmark_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    draw_figure1()
    draw_figure2()
    print("Done!")
