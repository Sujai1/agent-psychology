"""Frontier classification: predict P(frontier) instead of P(success).

Standard difficulty predictors estimate task difficulty b and derive P(success)
via sigmoid(theta - b_hat). This module reformulates the problem as binary
classification: given (agent_ability, task_features), is this (agent, task) pair
in the frontier difficulty band?

The frontier band is defined as the region where the IRT-predicted success
probability falls within [frontier_lo, frontier_hi] (default [0.3, 0.7]). Tasks
in this band are neither too easy nor too hard for the agent -- they are at the
"frontier" of the agent's ability.

predict_probability() returns P(frontier), NOT P(success). This is a different
semantic meaning from other CVPredictor implementations. Evaluation code must
account for this -- P(frontier) measures how likely a pair is to be informative
for distinguishing agent ability, not how likely the agent is to succeed.

Implements the CVPredictor protocol (fit / predict_probability) for use with
the cross-validation framework in cross_validation.py.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.feature_source import TaskFeatureSource


class FrontierClassifier:
    """Predicts P(frontier | agent, task) from [theta, task_features].

    Instead of predicting difficulty and converting to P(success), this directly
    classifies whether an (agent, task) pair falls in the frontier band where
    the IRT-predicted success probability is in [frontier_lo, frontier_hi].

    The frontier band identifies pairs where the task is neither trivially easy
    nor impossibly hard for the agent -- the most informative region for
    distinguishing agent ability.

    Training labels are derived from fold-specific IRT parameters:
        train_p = sigmoid(theta_train - b_train)
        label = 1 if frontier_lo <= train_p <= frontier_hi else 0

    These labels are an approximation of the oracle frontier (since fold-specific
    IRT has noisier estimates than full IRT), but this is the correct approach:
    we only use information available at training time.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        frontier_band: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        """Initialize the frontier classifier.

        Args:
            source: TaskFeatureSource providing task feature vectors.
            frontier_band: (lo, hi) defining the frontier success probability
                range. Pairs with IRT P(success) in [lo, hi] are labeled as
                frontier (positive class).

        Raises:
            ValueError: If frontier_band bounds are invalid.
        """
        if not (0.0 <= frontier_band[0] < frontier_band[1] <= 1.0):
            raise ValueError(
                f"frontier_band must satisfy 0 <= lo < hi <= 1, "
                f"got {frontier_band}"
            )

        self._source = source
        self._frontier_lo = frontier_band[0]
        self._frontier_hi = frontier_band[1]

        # Model state (set after fit)
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[GradientBoostingClassifier] = None
        self._is_fitted: bool = False

        # Cache for test task features (populated lazily in predict_probability)
        self._cached_test_features: Optional[Dict[str, np.ndarray]] = None

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the model on all (agent, train_task) observations.

        For each agent-task pair with a recorded response, creates a training
        row with features [theta_agent, task_feature_1, ..., task_feature_d]
        and a binary frontier label derived from IRT parameters.

        Class imbalance handling: frontier pairs are typically ~15-20% of the
        data. We compute sample weights inversely proportional to class
        frequency to ensure the model doesn't simply predict the majority class.

        Args:
            data: ExperimentData with responses and fold-specific IRT parameters.
            train_task_ids: Task IDs to train on.

        Raises:
            ValueError: If no training observations are found.
        """
        # Get task features for training tasks
        X_tasks = self._source.get_features(train_task_ids)
        task_id_to_idx = {tid: i for i, tid in enumerate(train_task_ids)}

        # Build training rows: one per (agent, task) observation
        rows: List[np.ndarray] = []
        labels: List[int] = []

        agents = data.get_all_agents()
        for agent_id in agents:
            theta = float(data.train_abilities.loc[agent_id, "ability"])
            agent_responses = data.responses.get(agent_id, {})

            for task_id in train_task_ids:
                if task_id not in agent_responses:
                    continue

                # Only use tasks that have IRT difficulty estimates
                if task_id not in data.train_items.index:
                    continue

                b = float(data.train_items.loc[task_id, "b"])

                # Compute IRT success probability from train parameters
                train_p = float(expit(theta - b))

                # Binary frontier label
                is_frontier = int(
                    self._frontier_lo <= train_p <= self._frontier_hi
                )

                task_idx = task_id_to_idx[task_id]
                task_feats = X_tasks[task_idx]

                # Feature vector: [theta, task_feature_1, ..., task_feature_d]
                row = np.concatenate([[theta], task_feats])
                rows.append(row)
                labels.append(is_frontier)

        if not rows:
            raise ValueError(
                "No training observations found. Check that train_task_ids "
                "have responses and IRT parameters."
            )

        X_train = np.vstack(rows)
        y_train = np.array(labels, dtype=int)

        # Compute sample weights inversely proportional to class frequency
        n_frontier = int(y_train.sum())
        n_non_frontier = len(y_train) - n_frontier

        if n_frontier == 0 or n_non_frontier == 0:
            raise ValueError(
                f"All training labels are the same class "
                f"(frontier={n_frontier}, non-frontier={n_non_frontier}). "
                f"Cannot train a classifier. Check frontier_band={self._frontier_lo, self._frontier_hi}."
            )

        weight_frontier = len(y_train) / (2.0 * n_frontier)
        weight_non_frontier = len(y_train) / (2.0 * n_non_frontier)
        sample_weights = np.where(
            y_train == 1, weight_frontier, weight_non_frontier
        )

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        # Train gradient boosting classifier
        self._model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X_scaled, y_train, sample_weight=sample_weights)

        self._is_fitted = True
        # Clear cached test features from any previous fold
        self._cached_test_features = None

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict probability that this (agent, task) pair is in the frontier band.

        NOTE: This returns P(frontier), NOT P(success). The frontier probability
        indicates how likely this pair is to fall in the informative difficulty
        region [frontier_lo, frontier_hi].

        On first call, caches task features for all test tasks to avoid
        repeated feature lookups.

        Args:
            data: ExperimentData for accessing agent abilities.
            agent_id: The agent to predict for.
            task_id: The task to predict for.

        Returns:
            Predicted probability that this pair is in the frontier band (0 to 1).

        Raises:
            RuntimeError: If called before fit().
            ValueError: If agent_id is not in train_abilities or task_id is
                not in test tasks.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FrontierClassifier must be fit before calling predict_probability()"
            )

        # Lazily cache test task features on first call
        if self._cached_test_features is None:
            test_tasks = data.test_tasks
            X_test = self._source.get_features(test_tasks)
            self._cached_test_features = {
                tid: X_test[i] for i, tid in enumerate(test_tasks)
            }

        if task_id not in self._cached_test_features:
            raise ValueError(
                f"Task {task_id} not found in cached test features. "
                f"It may not be in data.test_tasks."
            )

        if agent_id not in data.train_abilities.index:
            raise ValueError(f"Agent {agent_id} not found in train_abilities")

        theta = float(data.train_abilities.loc[agent_id, "ability"])
        task_feats = self._cached_test_features[task_id]

        # Build feature vector and scale
        x = np.concatenate([[theta], task_feats]).reshape(1, -1)
        x_scaled = self._scaler.transform(x)

        # Return probability of class 1 (frontier)
        prob = self._model.predict_proba(x_scaled)[0, 1]
        return float(prob)
