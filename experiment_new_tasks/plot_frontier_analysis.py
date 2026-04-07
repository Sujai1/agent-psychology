"""Plots for frontier difficulty identification analysis.

Generates:
1. Calibration plot: predicted p vs actual success rate per predictor
2. Per-band AUC bar chart: AUC within each oracle difficulty band

Usage:
    python -m experiment_new_tasks.plot_frontier_analysis
    python -m experiment_new_tasks.plot_frontier_analysis --results output/frontier_analysis/frontier_analysis.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from experiment_new_tasks.frontier_analysis import (
    DIFFICULTY_BANDS,
    FRONTIER_BAND,
    FrontierAnalysisResult,
)


# Consistent predictor colors
PREDICTOR_COLORS = {
    "oracle": "#2ca02c",
    "embedding": "#1f77b4",
    "llm_judge": "#ff7f0e",
    "grouped": "#d62728",
    "constant_baseline": "#9e9e9e",
    "weighted_ridge": "#17becf",
    "uncertainty": "#bcbd22",
    "direct": "#e377c2",
    "frontier_classifier": "#7f7f7f",
}

PREDICTOR_DISPLAY = {
    "oracle": "Oracle",
    "embedding": "Embedding",
    "llm_judge": "LLM Judge",
    "grouped": "Combined",
    "constant_baseline": "Baseline",
    "weighted_ridge": "Weighted Ridge",
    "uncertainty": "Uncertainty",
    "direct": "Direct (GBM)",
    "frontier_classifier": "Frontier Clf",
}

PREDICTOR_ORDER = [
    "constant_baseline", "embedding", "llm_judge", "grouped",
    "weighted_ridge", "uncertainty", "direct", "frontier_classifier",
    "oracle",
]


def plot_calibration(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
    output_dir: Path,
) -> None:
    """Plot calibration curves for all datasets.

    One subplot per dataset, one line per predictor.
    """
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    n_cols = min(n_datasets, 2)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for idx, dataset_name in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        predictor_results = all_results[dataset_name]

        # Diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

        # Shade frontier band
        flo, fhi = FRONTIER_BAND
        ax.axvspan(flo, fhi, alpha=0.08, color="orange", label="Frontier band")

        for pred_key in PREDICTOR_ORDER:
            if pred_key not in predictor_results:
                continue
            result = predictor_results[pred_key]
            color = PREDICTOR_COLORS.get(pred_key, "#333333")
            display = PREDICTOR_DISPLAY.get(pred_key, pred_key)

            # Filter to bins with data
            bins = result.calibration_bins
            predicted = result.calibration_predicted
            actual = result.calibration_actual
            counts = result.calibration_counts

            mask = [c > 0 for c in counts]
            x = [predicted[i] for i in range(len(bins)) if mask[i]]
            y = [actual[i] for i in range(len(bins)) if mask[i]]

            ax.plot(x, y, "o-", color=color, label=display, markersize=5, linewidth=1.5)

        ax.set_xlabel("Predicted probability", fontsize=12)
        ax.set_ylabel("Actual success rate", fontsize=12)
        ax.set_title(dataset_name, fontsize=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal")

    # Hide unused subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Calibration: Predicted Probability vs Actual Success Rate", fontsize=15, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_curves.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_band_aucs(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
    output_dir: Path,
) -> None:
    """Plot per-band AUC bar chart for all datasets.

    One subplot per dataset, grouped bars by predictor.
    """
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    n_cols = min(n_datasets, 2)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    band_labels = [f"[{lo:.1f},{hi:.1f})" for lo, hi in DIFFICULTY_BANDS]

    for idx, dataset_name in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        predictor_results = all_results[dataset_name]

        # Get predictors that exist in this dataset
        pred_keys = [k for k in PREDICTOR_ORDER if k in predictor_results]
        n_preds = len(pred_keys)
        n_bands = len(DIFFICULTY_BANDS)

        x = np.arange(n_bands)
        bar_width = 0.8 / max(n_preds, 1)

        for j, pred_key in enumerate(pred_keys):
            result = predictor_results[pred_key]
            color = PREDICTOR_COLORS.get(pred_key, "#333333")
            display = PREDICTOR_DISPLAY.get(pred_key, pred_key)

            aucs = []
            for band in DIFFICULTY_BANDS:
                auc = result.band_aucs.get(band)
                aucs.append(auc if auc is not None else 0.0)

            offset = (j - n_preds / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, aucs, bar_width,
                color=color, label=display, edgecolor="white", linewidth=0.5,
            )

            # Mark missing AUCs
            for k, band in enumerate(DIFFICULTY_BANDS):
                if result.band_aucs.get(band) is None:
                    ax.text(
                        x[k] + offset, 0.02, "n/a",
                        ha="center", va="bottom", fontsize=7, color="gray",
                    )

        # Highlight frontier band
        frontier_idx = next(
            i for i, (lo, hi) in enumerate(DIFFICULTY_BANDS) if lo == 0.4
        )
        ax.axvspan(
            frontier_idx - 0.5, frontier_idx + 0.5,
            alpha=0.08, color="orange", zorder=0,
        )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="Random (0.5)")
        ax.set_xlabel("Oracle difficulty band", fontsize=12)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title(dataset_name, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.08),
                  ncol=min(n_preds + 1, 5), frameon=False, columnspacing=1.0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.suptitle("Per-Band AUC: Discriminative Power by Difficulty Region", fontsize=15, y=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "band_aucs.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot frontier analysis results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("output/frontier_analysis/frontier_analysis.json"),
        help="Path to saved results JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/frontier_analysis/plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # If results JSON exists, load from it; otherwise run the analysis
    if args.results.exists():
        print(f"Loading results from {args.results}")
        with open(args.results) as f:
            raw = json.load(f)

        # Reconstruct FrontierAnalysisResult objects from JSON
        all_results: Dict[str, Dict[str, FrontierAnalysisResult]] = {}
        for dataset_name, predictor_data in raw.items():
            all_results[dataset_name] = {}
            for pred_key, data in predictor_data.items():
                # Reconstruct band_aucs with tuple keys
                band_aucs = {}
                band_counts = {}
                for band_str, auc in data["band_aucs"].items():
                    lo, hi = _parse_band_str(band_str)
                    band_aucs[(lo, hi)] = auc
                for band_str, n in data["band_counts"].items():
                    lo, hi = _parse_band_str(band_str)
                    band_counts[(lo, hi)] = n

                all_results[dataset_name][pred_key] = FrontierAnalysisResult(
                    predictor_name=data["predictor_name"],
                    dataset_name=data["dataset_name"],
                    band_aucs=band_aucs,
                    band_counts=band_counts,
                    frontier_precision=data["frontier_precision"],
                    frontier_recall=data["frontier_recall"],
                    frontier_n_predicted=data["frontier_n_predicted"],
                    frontier_n_true=data["frontier_n_true"],
                    calibration_bins=data["calibration_bins"],
                    calibration_predicted=data["calibration_predicted"],
                    calibration_actual=data["calibration_actual"],
                    calibration_counts=data["calibration_counts"],
                    calibration_ece=data["calibration_ece"],
                    calibration_frontier_ece=data["calibration_frontier_ece"],
                    overall_auc=data["overall_auc"],
                    n_total_pairs=data["n_total_pairs"],
                )
    else:
        print(f"Results not found at {args.results}. Run run_frontier_analysis.py first.")
        return

    plot_calibration(all_results, args.output_dir)
    plot_band_aucs(all_results, args.output_dir)


def _parse_band_str(band_str: str) -> tuple:
    """Parse '[0.0,0.2)' -> (0.0, 0.2)."""
    cleaned = band_str.strip("[]() ")
    lo, hi = cleaned.split(",")
    return float(lo), float(hi)


if __name__ == "__main__":
    main()
