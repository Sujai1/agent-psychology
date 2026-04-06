"""Plot feature source ablation bar graph (Table 3).

Each dataset gets 3 bars: Baseline, Embedding, and LLM-as-a-Judge.
Stacked segments within each bar use a consistent color per feature source.

Usage:
    python -m experiment_new_tasks.plot_information_ablation
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Data from paper Table 2 (baseline) and Table 3 (ablation) ──────────────
DATASETS = ["SWE-bench\nVerified", "SWE-bench\nPro", "GSO", "Terminal-Bench\n2.0"]

BASELINE = [0.718, 0.657, 0.714, 0.734]

# Embedding: cumulative AUC at each info level
EMBED_LEVELS = [
    [0.758, 0.741, 0.677, 0.782],  # Problem
    [0.824, 0.755, 0.762, 0.817],  # + Solution
]

# LLM Judge: cumulative AUC at each info level
JUDGE_LEVELS = [
    [0.787, 0.718, 0.726, 0.799],  # Problem
    [0.798, 0.737, 0.727, 0.807],  # + Repository State
    [0.834, 0.749, 0.725, 0.807],  # + Tests
    [0.848, 0.750, 0.797, 0.810],  # + Solution
]

# ── Colors: one per feature source ─────────────────────────────────────────
COLOR_BASELINE = "#9e9e9e"
COLOR_PROBLEM = "#4a90d9"
COLOR_REPO = "#e8833a"
COLOR_TESTS = "#59a14f"
COLOR_SOLUTION = "#b07aa1"

EMBED_COLORS = [COLOR_PROBLEM, COLOR_SOLUTION]
JUDGE_COLORS = [COLOR_PROBLEM, COLOR_REPO, COLOR_TESTS, COLOR_SOLUTION]

BAR_WIDTH = 0.22
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 10

OUT = Path("output/information_ablation_barplot.png")


def compute_segments(
    levels: List[List[float]],
) -> List[Tuple[List[float], List[float]]]:
    """Given cumulative AUC values at each level, return (bottoms, heights) per segment.

    Handles non-monotonic levels by clamping negative increments to 0.
    """
    n = len(levels[0])
    result = []
    current_top = [0.0] * n
    for level_vals in levels:
        heights = [max(0.0, level_vals[j] - current_top[j]) for j in range(n)]
        result.append((list(current_top), heights))
        current_top = [max(current_top[j], level_vals[j]) for j in range(n)]
    return result


def main() -> None:
    x = np.arange(len(DATASETS))
    offsets = np.array([-1, 0, 1]) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # ── Baseline bar ────────────────────────────────────────────────────────
    ax.bar(
        x + offsets[0], BASELINE, BAR_WIDTH,
        color=COLOR_BASELINE, edgecolor="white", linewidth=0.5,
    )

    # ── Embedding bar (stacked segments) ────────────────────────────────────
    embed_segs = compute_segments(EMBED_LEVELS)
    for (bottoms, heights), color in zip(embed_segs, EMBED_COLORS):
        ax.bar(
            x + offsets[1], heights, BAR_WIDTH,
            bottom=bottoms, color=color, edgecolor="white", linewidth=0.5,
        )

    # ── LLM Judge bar (stacked segments) ────────────────────────────────────
    judge_segs = compute_segments(JUDGE_LEVELS)
    for (bottoms, heights), color in zip(judge_segs, JUDGE_COLORS):
        ax.bar(
            x + offsets[2], heights, BAR_WIDTH,
            bottom=bottoms, color=color, edgecolor="white", linewidth=0.5,
        )

    # ── Sub-labels under each bar ─────────────────────────────────────────
    sub_labels = ["Base", "Emb", "Judge"]
    for j in range(len(DATASETS)):
        for k, label in enumerate(sub_labels):
            ax.text(
                x[j] + offsets[k], 0.49, label,
                ha="center", va="top", fontsize=8, color="black",
                fontweight="bold",
            )

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_ylabel("AUC-ROC", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0.47, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BASELINE, edgecolor="white", label="Baseline"),
        mpatches.Patch(facecolor=COLOR_PROBLEM, edgecolor="white", label="Problem Statement"),
        mpatches.Patch(facecolor=COLOR_REPO, edgecolor="white", label="+ Repo State"),
        mpatches.Patch(facecolor=COLOR_TESTS, edgecolor="white", label="+ Tests"),
        mpatches.Patch(facecolor=COLOR_SOLUTION, edgecolor="white", label="+ Solution"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        fontsize=LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()

    # Draw bracket after layout is finalized, using display coords → figure coords
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()

    texts = leg.get_texts()
    top_bb = texts[2].get_window_extent(renderer)  # "+ Repo State"
    bot_bb = texts[4].get_window_extent(renderer)  # "+ Solution"
    leg_bb = leg.get_window_extent(renderer)

    # Convert to figure coordinates
    top_y = inv_fig.transform((0, top_bb.y1))[1]
    bot_y = inv_fig.transform((0, bot_bb.y0))[1]
    brace_x = inv_fig.transform((leg_bb.x1 + 4, 0))[0]
    tick_len = 0.006
    mid_y = (top_y + bot_y) / 2

    bracket_style = dict(color="black", linewidth=1.2, clip_on=False,
                         transform=fig.transFigure)
    fig.lines.extend([
        plt.Line2D([brace_x, brace_x], [bot_y, top_y], **bracket_style),
        plt.Line2D([brace_x, brace_x - tick_len], [top_y, top_y], **bracket_style),
        plt.Line2D([brace_x, brace_x - tick_len], [bot_y, bot_y], **bracket_style),
    ])

    fig.text(
        brace_x + 0.008, mid_y, "Agentic\nartifacts",
        fontsize=LEGEND_SIZE, va="center", ha="left", fontstyle="italic",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15)
    print(OUT)


if __name__ == "__main__":
    main()
