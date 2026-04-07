"""Frontier difficulty identification analysis.

Evaluates whether IRT difficulty predictors can accurately identify
medium-difficulty tasks — those where agents succeed ~40-60% of the time.

This module provides:
- PairRecord: raw (agent, task, pred_p, oracle_p, actual) data
- collect_fold_records(): run one fold and capture all prediction pairs
- compute_band_aucs(): AUC within oracle_p difficulty bands
- compute_frontier_precision_recall(): precision/recall for frontier identification
- compute_calibration(): binned calibration metrics
- analyze_predictor(): full analysis across all folds
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

from experiment_new_tasks.cross_validation import CVPredictor
from experiment_new_tasks.dataset import ExperimentData


DIFFICULTY_BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
FRONTIER_BAND = (0.3, 0.7)
MIN_PAIRS_FOR_AUC = 30


@dataclass
class PairRecord:
    """A single (agent, task) prediction record."""
    agent_id: str
    task_id: str
    pred_p: float
    oracle_p: float
    actual: int


@dataclass
class FrontierAnalysisResult:
    """Full frontier analysis results for one predictor on one dataset."""
    predictor_name: str
    dataset_name: str

    # Per-band metrics
    band_aucs: Dict[Tuple[float, float], Optional[float]]
    band_counts: Dict[Tuple[float, float], int]

    # Frontier identification
    frontier_precision: Optional[float]
    frontier_recall: Optional[float]
    frontier_n_predicted: int
    frontier_n_true: int

    # Calibration
    calibration_bins: List[float]
    calibration_predicted: List[float]
    calibration_actual: List[float]
    calibration_counts: List[int]
    calibration_ece: float
    calibration_frontier_ece: Optional[float]

    # Overall AUC (sanity check)
    overall_auc: Optional[float]

    # Total pair count
    n_total_pairs: int

    # Raw records (for downstream plotting)
    all_records: List[PairRecord] = field(default_factory=list, repr=False)


def collect_fold_records(
    predictor: CVPredictor,
    data: ExperimentData,
    train_tasks: List[str],
    test_tasks: List[str],
) -> List[PairRecord]:
    """Run predictor on one fold and collect all prediction pairs.

    Replicates the evaluation loop from cross_validation._run_single_fold
    but captures raw data instead of computing AUC directly.
    """
    predictor.fit(data, train_tasks)

    records = []
    for task_id in test_tasks:
        # Compute oracle difficulty for this task
        if task_id not in data.full_items.index:
            continue
        beta_oracle = data.full_items.loc[task_id, "b"]

        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            pred_p = predictor.predict_probability(data, agent_id, task_id)

            theta_oracle = data.full_abilities.loc[agent_id, "ability"]
            oracle_p = float(sigmoid(theta_oracle - beta_oracle))

            actual = data.responses[agent_id][task_id]

            records.append(PairRecord(
                agent_id=agent_id,
                task_id=task_id,
                pred_p=pred_p,
                oracle_p=oracle_p,
                actual=int(actual),
            ))

    return records


def compute_band_aucs(
    records: List[PairRecord],
    bands: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Dict[Tuple[float, float], Optional[float]], Dict[Tuple[float, float], int]]:
    """Compute AUC within each oracle_p difficulty band.

    Returns:
        (band_aucs, band_counts) where band_aucs maps band -> AUC or None.
    """
    if bands is None:
        bands = DIFFICULTY_BANDS

    band_aucs = {}
    band_counts = {}

    for lo, hi in bands:
        # Filter to records in this band
        band_records = [
            r for r in records
            if lo <= r.oracle_p < hi or (hi == 1.0 and r.oracle_p == 1.0)
        ]
        band_counts[(lo, hi)] = len(band_records)

        if len(band_records) < MIN_PAIRS_FOR_AUC:
            band_aucs[(lo, hi)] = None
            continue

        y_true = [r.actual for r in band_records]
        y_scores = [r.pred_p for r in band_records]

        if len(set(y_true)) < 2:
            band_aucs[(lo, hi)] = None
            continue

        band_aucs[(lo, hi)] = float(roc_auc_score(y_true, y_scores))

    return band_aucs, band_counts


def compute_frontier_precision_recall(
    records: List[PairRecord],
    frontier: Tuple[float, float] = FRONTIER_BAND,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """Compute precision and recall for frontier task identification.

    Precision: P(oracle_p in frontier | pred_p in frontier)
    Recall: P(pred_p in frontier | oracle_p in frontier)

    Returns:
        (precision, recall, n_predicted_frontier, n_true_frontier)
    """
    lo, hi = frontier

    predicted_frontier = [r for r in records if lo <= r.pred_p <= hi]
    true_frontier = [r for r in records if lo <= r.oracle_p <= hi]

    n_predicted = len(predicted_frontier)
    n_true = len(true_frontier)

    if n_predicted == 0:
        precision = None
    else:
        correct = sum(1 for r in predicted_frontier if lo <= r.oracle_p <= hi)
        precision = correct / n_predicted

    if n_true == 0:
        recall = None
    else:
        found = sum(1 for r in true_frontier if lo <= r.pred_p <= hi)
        recall = found / n_true

    return precision, recall, n_predicted, n_true


def compute_calibration(
    records: List[PairRecord],
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[float], List[int], float, Optional[float]]:
    """Compute calibration metrics by binning predicted probabilities.

    Returns:
        (bin_centers, mean_predicted, actual_rate, counts, ece, frontier_ece)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    mean_predicted = []
    actual_rate = []
    counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_records = [
            r for r in records
            if lo <= r.pred_p < hi or (i == n_bins - 1 and r.pred_p == hi)
        ]

        bin_centers.append((lo + hi) / 2)
        counts.append(len(bin_records))

        if len(bin_records) == 0:
            mean_predicted.append((lo + hi) / 2)
            actual_rate.append(0.0)
        else:
            mean_predicted.append(np.mean([r.pred_p for r in bin_records]))
            actual_rate.append(np.mean([r.actual for r in bin_records]))

    # ECE: weighted average of |predicted - actual| across bins
    total = sum(counts)
    if total > 0:
        ece = sum(
            counts[i] * abs(mean_predicted[i] - actual_rate[i])
            for i in range(n_bins)
            if counts[i] > 0
        ) / total
    else:
        ece = 0.0

    # Frontier ECE: only bins in [0.3, 0.7] range
    frontier_lo, frontier_hi = FRONTIER_BAND
    frontier_bins = [
        i for i in range(n_bins)
        if bin_centers[i] >= frontier_lo and bin_centers[i] <= frontier_hi
        and counts[i] > 0
    ]
    if frontier_bins:
        frontier_total = sum(counts[i] for i in frontier_bins)
        frontier_ece = sum(
            counts[i] * abs(mean_predicted[i] - actual_rate[i])
            for i in frontier_bins
        ) / frontier_total if frontier_total > 0 else None
    else:
        frontier_ece = None

    return bin_centers, mean_predicted, actual_rate, counts, ece, frontier_ece


def analyze_predictor(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    predictor_name: str,
    dataset_name: str,
    verbose: bool = True,
) -> FrontierAnalysisResult:
    """Full frontier analysis for one predictor across all folds.

    Pools records across all folds, then computes all metrics.
    """
    all_records = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        data = load_fold_data(train_tasks, test_tasks, fold_idx)
        fold_records = collect_fold_records(predictor, data, train_tasks, test_tasks)
        all_records.extend(fold_records)

        if verbose:
            print(f"      Fold {fold_idx + 1}: {len(fold_records)} pairs")

    if verbose:
        print(f"      Total: {len(all_records)} pairs")

    # Overall AUC (sanity check)
    y_true = [r.actual for r in all_records]
    y_scores = [r.pred_p for r in all_records]
    if len(set(y_true)) >= 2:
        overall_auc = float(roc_auc_score(y_true, y_scores))
    else:
        overall_auc = None

    # Per-band AUC
    band_aucs, band_counts = compute_band_aucs(all_records)

    # Frontier precision/recall
    precision, recall, n_pred, n_true = compute_frontier_precision_recall(all_records)

    # Calibration
    bin_centers, mean_pred, act_rate, counts, ece, frontier_ece = compute_calibration(
        all_records
    )

    return FrontierAnalysisResult(
        predictor_name=predictor_name,
        dataset_name=dataset_name,
        band_aucs=band_aucs,
        band_counts=band_counts,
        frontier_precision=precision,
        frontier_recall=recall,
        frontier_n_predicted=n_pred,
        frontier_n_true=n_true,
        calibration_bins=bin_centers,
        calibration_predicted=mean_pred,
        calibration_actual=act_rate,
        calibration_counts=counts,
        calibration_ece=ece,
        calibration_frontier_ece=frontier_ece,
        overall_auc=overall_auc,
        n_total_pairs=len(all_records),
        all_records=all_records,
    )
