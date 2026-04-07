"""Frontier-weighted Ridge predictor for task difficulty.

Same as the standard Ridge regression approach (predict IRT difficulty b from
task features), but with sample weights that emphasize frontier tasks — those
with ~50% empirical solve rate. The standard Ridge treats all tasks equally in
the MSE loss, which means the extremes (very easy / very hard) dominate the
fit. By upweighting frontier tasks, the model focuses on getting those right
even at the expense of extreme tasks.

The weight formula is: w(p) = 4 * p * (1 - p)
    - p = 0.50  =>  w = 1.00  (full weight)
    - p = 0.10  =>  w = 0.36
    - p = 0.01  =>  w = 0.04  (nearly zero)

Implements the CVPredictor protocol from cross_validation.py.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.feature_source import TaskFeatureSource


class WeightedRidgePredictor:
    """Ridge predictor with sample weights emphasizing frontier tasks.

    Pipeline: features -> StandardScaler -> RidgeCV(sample_weight) -> predict

    Sample weights are derived from each task's empirical solve rate across
    all agents in the response matrix. Tasks near 50% solve rate receive
    weight ~1.0; tasks near 0% or 100% receive weight near 0.

    Implements the CVPredictor protocol:
        fit(data, train_task_ids) -> None
        predict_probability(data, agent_id, task_id) -> float
    """

    DEFAULT_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    def __init__(
        self,
        source: TaskFeatureSource,
        alphas: Optional[List[float]] = None,
    ):
        """Initialize the frontier-weighted Ridge predictor.

        Args:
            source: TaskFeatureSource that provides features for tasks.
            alphas: List of alpha values for RidgeCV cross-validation.
                Defaults to [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0].
        """
        self.source = source
        self.alphas = alphas if alphas is not None else self.DEFAULT_RIDGE_ALPHAS

        # Model state (set after fit)
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[RidgeCV] = None
        self._is_fitted: bool = False
        self._predicted_difficulties: Dict[str, float] = {}

    def _compute_frontier_weights(
        self,
        task_ids: List[str],
        responses: Dict[str, Dict[str, int]],
    ) -> np.ndarray:
        """Compute sample weights based on empirical solve rate.

        Weight formula: w = 4 * p * (1 - p), which peaks at p=0.5.

        Args:
            task_ids: Task IDs to compute weights for.
            responses: Full response matrix {agent_id: {task_id: 0|1}}.

        Returns:
            Array of sample weights, one per task.

        Raises:
            ValueError: If a task has zero total responses.
        """
        weights = np.empty(len(task_ids), dtype=np.float64)

        for i, task_id in enumerate(task_ids):
            n_solved = 0
            n_total = 0
            for agent_responses in responses.values():
                if task_id in agent_responses:
                    n_total += 1
                    if agent_responses[task_id] == 1:
                        n_solved += 1

            if n_total == 0:
                raise ValueError(
                    f"Task '{task_id}' has zero responses in the response matrix. "
                    f"Cannot compute empirical solve rate."
                )

            p_empirical = n_solved / n_total
            weights[i] = 4.0 * p_empirical * (1.0 - p_empirical)

        return weights

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the predictor on training data with frontier weighting.

        Args:
            data: ExperimentData containing responses, IRT parameters, etc.
            train_task_ids: List of task IDs to train on.
        """
        # Clear cached predictions from previous fold
        self._predicted_difficulties = {}

        # Get features and ground truth difficulties
        X = self.source.get_features(train_task_ids)
        y = data.train_items.loc[train_task_ids, "b"].values

        # Compute frontier-based sample weights
        sample_weights = self._compute_frontier_weights(
            train_task_ids, data.responses
        )

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit RidgeCV with sample weights
        self._model = RidgeCV(alphas=self.alphas, cv=5)
        self._model.fit(X_scaled, y, sample_weight=sample_weights)

        self._is_fitted = True

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula: sigmoid(theta - b_hat).

        Lazily predicts difficulties for all test tasks on first call.

        Args:
            data: ExperimentData for accessing agent abilities and test tasks.
            agent_id: The agent whose ability theta is used.
            task_id: The task whose difficulty b_hat is predicted.

        Returns:
            Predicted probability of success (0 to 1).

        Raises:
            RuntimeError: If called before fit().
            ValueError: If task_id has no predicted difficulty or agent_id
                is not found in abilities.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Predictor must be fit before calling predict_probability()"
            )

        # Lazily predict difficulties for all test tasks
        if task_id not in self._predicted_difficulties:
            test_tasks = data.test_tasks
            X = self.source.get_features(test_tasks)
            X_scaled = self._scaler.transform(X)
            predictions = self._model.predict(X_scaled)
            self._predicted_difficulties.update(
                {t: float(p) for t, p in zip(test_tasks, predictions)}
            )

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]
        theta = data.train_abilities.loc[agent_id, "ability"]
        return float(sigmoid(theta - beta))
