#!/usr/bin/env python3
"""Generate visualization for Head Agent category reduction analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_PATH = "docs/category_reduction_analysis.png"

# --- Data from 3DSRBench 200-sample test ---
FINE_CATS = [
    ("location_above",                       90.3, 31, "vertical"),
    ("height_higher",                         0.0, 26, "vertical"),
    ("location_closer_to_camera",           100.0, 30, "camera_dist"),
    ("multi_object_closer_to",               71.4, 14, "camera_dist"),
    ("location_next_to",                    100.0, 14, "adjacency"),
    ("orientation_on_the_left",              14.3, 14, "orientation"),
    ("orientation_in_front_of",              55.6,  9, "orientation"),
    ("orientation_viewpoint",                60.0, 10, "orientation"),
    ("multi_object_facing",                   0.0,  8, "orientation"),
    ("multi_object_same_direction",          75.0, 12, "orientation"),
    ("multi_object_viewpoint_towards_obj",    6.2, 16, "orientation"),
    ("multi_object_parallel",               100.0, 16, "alignment"),
]

GROUPED = [
    ("vertical",    94.7, 57),
    ("camera_dist", 100.0, 44),
    ("adjacency",   100.0, 14),
    ("orientation", 100.0, 69),
    ("alignment",   100.0, 16),
]

GROUP_COLORS = {
    "vertical":    "#4C72B0",
    "camera_dist": "#55A868",
    "adjacency":   "#C44E52",
    "orientation": "#8172B3",
    "alignment":   "#CCB974",
}

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Head Agent Category Classification — 3DSRBench\n"
    "Why 12 fine-grained categories should be reduced to 5 groups",
    fontsize=18, fontweight="bold", y=0.98,
)

# ======================================================================
# Panel 1: Fine-grained 12 categories (horizontal bar)
# ======================================================================
ax1 = fig.add_subplot(2, 2, 1)
names = [c[0] for c in FINE_CATS]
accs = [c[1] for c in FINE_CATS]
counts = [c[2] for c in FINE_CATS]
groups = [c[3] for c in FINE_CATS]
colors = [GROUP_COLORS[g] for g in groups]

y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, accs, color=colors, edgecolor="white", height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlim(0, 115)
ax1.set_xlabel("Accuracy (%)")
ax1.set_title("Fine-Grained (12 categories): 60.5%", fontsize=13, fontweight="bold")
ax1.invert_yaxis()
ax1.axvline(x=60.5, color="red", linestyle="--", alpha=0.7, label="Overall 60.5%")

for i, (acc, n) in enumerate(zip(accs, counts)):
    ax1.text(acc + 1.5, i, f"{acc:.0f}% (n={n})", va="center", fontsize=8)

ax1.legend(loc="lower right", fontsize=9)

# ======================================================================
# Panel 2: Grouped 5 categories (horizontal bar)
# ======================================================================
ax2 = fig.add_subplot(2, 2, 2)
g_names = [g[0] for g in GROUPED]
g_accs = [g[1] for g in GROUPED]
g_counts = [g[2] for g in GROUPED]
g_colors = [GROUP_COLORS[g] for g, _, _ in GROUPED]

y_pos2 = np.arange(len(g_names))
ax2.barh(y_pos2, g_accs, color=g_colors, edgecolor="white", height=0.5)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(g_names, fontsize=11, fontweight="bold")
ax2.set_xlim(0, 115)
ax2.set_xlabel("Accuracy (%)")
ax2.set_title("Grouped (5 categories): 98.5%", fontsize=13, fontweight="bold")
ax2.invert_yaxis()
ax2.axvline(x=98.5, color="green", linestyle="--", alpha=0.7, label="Overall 98.5%")

for i, (acc, n) in enumerate(zip(g_accs, g_counts)):
    ax2.text(acc + 1.5, i, f"{acc:.0f}% (n={n})", va="center", fontsize=10)

ax2.legend(loc="lower right", fontsize=9)

# ======================================================================
# Panel 3: Side-by-side comparison
# ======================================================================
ax3 = fig.add_subplot(2, 2, 3)
labels = ["Fine-grained\n(12 cats)", "Grouped\n(5 cats)"]
values = [60.5, 98.5]
bar_colors = ["#E74C3C", "#2ECC71"]
bars3 = ax3.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")
ax3.set_ylim(0, 110)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Overall Accuracy Comparison", fontsize=13, fontweight="bold")

for bar, val in zip(bars3, values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val}%",
             ha="center", fontsize=16, fontweight="bold")

ax3.annotate(
    "+38.0pp",
    xy=(1, 98.5), xytext=(0.5, 80),
    fontsize=14, fontweight="bold", color="#2ECC71",
    arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
    ha="center",
)

# ======================================================================
# Panel 4: Key findings text
# ======================================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis("off")

findings = """KEY FINDINGS

Model: Qwen3-VL-4B  |  Benchmark: 3DSRBench  |  Samples: 200

Problem:
  12 fine-grained categories have heavy semantic overlap.
  The model correctly identifies the GROUP but picks
  the wrong sub-category within the group.

  Examples of within-group confusion:
    • height_higher → location_above       (26/26 = 100%)
    • multi_object_facing → other orientation cats
    • orientation_on_the_left → orientation_viewpoint

Evidence:
  Fine-grained accuracy:  60.5%  (121/200)
  Grouped accuracy:       98.5%  (197/200)
  Gap:                   +38.0 percentage points

  Groups with 100% accuracy:
    camera_dist, adjacency, orientation, alignment

  Only error: 3 vertical samples → UNKNOWN (answered
  the question instead of classifying)

Recommendation:
  Reduce 12 → 5 group-level categories for the
  Score Map and agent selection.
  Fine-grained distinctions add noise, not signal."""

ax4.text(0.05, 0.95, findings, transform=ax4.transAxes,
         fontsize=10, fontfamily="monospace",
         verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6"))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUTPUT_PATH}")
