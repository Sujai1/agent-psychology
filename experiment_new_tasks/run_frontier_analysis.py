#!/usr/bin/env python3
"""Frontier difficulty identification analysis.

Evaluates whether IRT difficulty predictors can accurately identify
medium-difficulty tasks (those where agents succeed ~40-60% of the time).

Usage:
    python -m experiment_new_tasks.run_frontier_analysis
    python -m experiment_new_tasks.run_frontier_analysis --datasets swebench_verified
    python -m experiment_new_tasks.run_frontier_analysis --datasets swebench_verified --plots
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_new_tasks.config import DATASET_DEFAULTS, ExperimentAConfig
from experiment_new_tasks.cross_validation import k_fold_split_tasks
from experiment_new_tasks.dataset import load_dataset_for_fold, _load_binary_responses
from experiment_new_tasks.dataset import filter_unsolved_tasks
from experiment_new_tasks.frontier_analysis import (
    DIFFICULTY_BANDS,
    FRONTIER_BAND,
    FrontierAnalysisResult,
    analyze_predictor,
)
from experiment_new_tasks.pipeline import build_cv_predictors

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ALL_DATASETS = ["swebench_verified", "swebench_pro", "gso", "terminalbench"]
OUTPUT_DIR = Path("output/frontier_analysis")


def run_frontier_analysis_for_dataset(
    dataset: str,
    k_folds: int = 5,
) -> Dict[str, FrontierAnalysisResult]:
    """Run frontier analysis for all predictors on one dataset."""
    config = ExperimentAConfig.for_dataset(dataset)

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    if config.exclude_unsolved:
        responses = _load_binary_responses(responses_path)
        all_task_ids, n_excluded = filter_unsolved_tasks(all_task_ids, responses)
        print(f"  Excluded {n_excluded} unsolved tasks ({len(all_task_ids)} remaining)")

    print(f"  Tasks: {len(all_task_ids)}, Folds: {k_folds}")

    # Generate folds (same seed as Table 2 for reproducibility)
    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    # Build fold data loader
    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            irt_cache_dir=ROOT / config.irt_cache_dir,
            exclude_unsolved=config.exclude_unsolved,
        )

    # Build predictors
    predictor_configs = build_cv_predictors(config, ROOT)

    # Run analysis for each predictor
    results: Dict[str, FrontierAnalysisResult] = {}
    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n  {i}. {pc.display_name}:")
        results[pc.name] = analyze_predictor(
            pc.predictor,
            folds,
            load_fold_data,
            predictor_name=pc.display_name,
            dataset_name=config.display_name,
            verbose=True,
        )
        r = results[pc.name]
        auc_str = f"{r.overall_auc:.4f}" if r.overall_auc else "N/A"
        print(f"      Overall AUC: {auc_str}")

    return results


def print_band_auc_table(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
) -> None:
    """Print per-band AUC table across datasets and predictors."""
    print("\n" + "=" * 90)
    print("PER-BAND AUC (by oracle difficulty band)")
    print("=" * 90)

    for dataset_name, predictor_results in all_results.items():
        print(f"\n{dataset_name}")
        print("-" * 90)

        # Header
        band_labels = [f"[{lo:.1f},{hi:.1f})" for lo, hi in DIFFICULTY_BANDS]
        header = f"{'Predictor':<30}" + "".join(f"{'  ' + bl:>14}" for bl in band_labels) + f"{'Overall':>10}"
        print(header)
        print("-" * 90)

        for name, result in predictor_results.items():
            row = f"{result.predictor_name:<30}"
            for band in DIFFICULTY_BANDS:
                auc = result.band_aucs.get(band)
                n = result.band_counts.get(band, 0)
                if auc is not None:
                    row += f"  {auc:.3f} ({n:>4})"
                else:
                    row += f"     -  ({n:>4})"
            auc_str = f"{result.overall_auc:.4f}" if result.overall_auc else "N/A"
            row += f"  {auc_str}"
            print(row)


def print_frontier_table(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
) -> None:
    """Print frontier identification precision/recall table."""
    lo, hi = FRONTIER_BAND
    print(f"\n{'=' * 90}")
    print(f"FRONTIER IDENTIFICATION (pred_p in [{lo},{hi}] vs oracle_p in [{lo},{hi}])")
    print("=" * 90)

    header = f"{'Dataset':<22} {'Predictor':<30} {'Prec':>6} {'Recall':>7} {'#Pred':>7} {'#True':>7} {'#Total':>7}"
    print(header)
    print("-" * 90)

    for dataset_name, predictor_results in all_results.items():
        for name, result in predictor_results.items():
            prec_str = f"{result.frontier_precision:.3f}" if result.frontier_precision is not None else "-"
            rec_str = f"{result.frontier_recall:.3f}" if result.frontier_recall is not None else "-"
            print(
                f"{dataset_name:<22} {result.predictor_name:<30} "
                f"{prec_str:>6} {rec_str:>7} "
                f"{result.frontier_n_predicted:>7} {result.frontier_n_true:>7} "
                f"{result.n_total_pairs:>7}"
            )


def print_calibration_table(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
) -> None:
    """Print calibration ECE table."""
    print(f"\n{'=' * 90}")
    print("CALIBRATION (Expected Calibration Error)")
    print("=" * 90)

    header = f"{'Dataset':<22} {'Predictor':<30} {'ECE':>8} {'Frontier ECE':>14}"
    print(header)
    print("-" * 90)

    for dataset_name, predictor_results in all_results.items():
        for name, result in predictor_results.items():
            fece_str = f"{result.calibration_frontier_ece:.4f}" if result.calibration_frontier_ece is not None else "-"
            print(
                f"{dataset_name:<22} {result.predictor_name:<30} "
                f"{result.calibration_ece:>8.4f} {fece_str:>14}"
            )


def save_results_json(
    all_results: Dict[str, Dict[str, FrontierAnalysisResult]],
    output_dir: Path,
) -> None:
    """Save results to JSON (excluding raw records for size)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for dataset_name, predictor_results in all_results.items():
        serializable[dataset_name] = {}
        for name, result in predictor_results.items():
            serializable[dataset_name][name] = {
                "predictor_name": result.predictor_name,
                "dataset_name": result.dataset_name,
                "overall_auc": result.overall_auc,
                "n_total_pairs": result.n_total_pairs,
                "band_aucs": {
                    f"[{lo},{hi})": auc
                    for (lo, hi), auc in result.band_aucs.items()
                },
                "band_counts": {
                    f"[{lo},{hi})": n
                    for (lo, hi), n in result.band_counts.items()
                },
                "frontier_precision": result.frontier_precision,
                "frontier_recall": result.frontier_recall,
                "frontier_n_predicted": result.frontier_n_predicted,
                "frontier_n_true": result.frontier_n_true,
                "calibration_ece": result.calibration_ece,
                "calibration_frontier_ece": result.calibration_frontier_ece,
                "calibration_bins": result.calibration_bins,
                "calibration_predicted": result.calibration_predicted,
                "calibration_actual": result.calibration_actual,
                "calibration_counts": result.calibration_counts,
            }

    path = output_dir / "frontier_analysis.json"
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Frontier difficulty identification analysis"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        help="Specific datasets to run (default: all)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots after analysis",
    )
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS

    print("=" * 60)
    print("FRONTIER DIFFICULTY IDENTIFICATION ANALYSIS")
    print("=" * 60)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"K-folds: {args.k_folds}")
    print(f"Frontier band: {FRONTIER_BAND}")
    print(f"Difficulty bands: {DIFFICULTY_BANDS}")

    all_results: Dict[str, Dict[str, FrontierAnalysisResult]] = {}

    for dataset in datasets:
        display_name = DATASET_DEFAULTS[dataset]["display_name"]
        print(f"\n{'=' * 60}")
        print(f"Dataset: {display_name}")
        print("=" * 60)

        results = run_frontier_analysis_for_dataset(
            dataset, k_folds=args.k_folds
        )
        all_results[display_name] = results

    # Print summary tables
    print_band_auc_table(all_results)
    print_frontier_table(all_results)
    print_calibration_table(all_results)

    # Save results
    save_results_json(all_results, ROOT / args.output_dir)

    # Generate plots if requested
    if args.plots:
        from experiment_new_tasks.plot_frontier_analysis import (
            plot_calibration,
            plot_band_aucs,
        )
        plot_dir = ROOT / args.output_dir / "plots"
        plot_calibration(all_results, plot_dir)
        plot_band_aucs(all_results, plot_dir)


if __name__ == "__main__":
    main()
