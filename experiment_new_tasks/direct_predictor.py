"""Direct prediction of agent success on tasks.

Instead of the paper's 2-stage pipeline:
    1. Predict difficulty b_hat from task features
    2. P(success) = sigmoid(theta - b_hat)

this module predicts P(success) directly from [agent_ability, task_features].
This removes the information bottleneck of compressing all task information
through a single scalar difficulty parameter b.

The model uses a GradientBoostingClassifier, which can capture non-linear
interactions between agent ability and task features (e.g., "strong agents
handle complex tasks but weak agents don't" is a natural interaction that
a linear model through a scalar bottleneck cannot express).
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.feature_source import TaskFeatureSource


class DirectPredictor:
    """Predicts P(success | agent, task) directly from [theta, task_features].

    Implements the CVPredictor protocol (fit / predict_probability) so it
    can be used with the cross-validation framework in cross_validation.py.

    Unlike difficulty-based predictors that compress task information into a
    single scalar b, this model has access to the full task feature vector
    when making predictions, allowing it to learn feature-specific
    interactions with agent ability.
    """

    def __init__(self, source: TaskFeatureSource) -> None:
        """Initialize the direct predictor.

        Args:
            source: TaskFeatureSource providing task feature vectors.
        """
        self._source = source

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
        and label 0 or 1.

        Args:
            data: ExperimentData with responses and IRT abilities.
            train_task_ids: Task IDs to train on.
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

                task_idx = task_id_to_idx[task_id]
                task_feats = X_tasks[task_idx]

                # Feature vector: [theta, task_feature_1, ..., task_feature_d]
                row = np.concatenate([[theta], task_feats])
                rows.append(row)
                labels.append(agent_responses[task_id])

        X_train = np.vstack(rows)
        y_train = np.array(labels, dtype=int)

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
        self._model.fit(X_scaled, y_train)

        self._is_fitted = True
        # Clear cached test features from any previous fold
        self._cached_test_features = None

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict probability of success for an (agent, task) pair.

        On first call, caches task features for all test tasks to avoid
        repeated feature lookups.

        Args:
            data: ExperimentData for accessing agent abilities.
            agent_id: The agent to predict for.
            task_id: The task to predict for.

        Returns:
            Predicted probability of success (0 to 1).

        Raises:
            RuntimeError: If called before fit().
            ValueError: If agent_id is not in train_abilities.
        """
        if not self._is_fitted:
            raise RuntimeError("DirectPredictor must be fit before calling predict_probability()")

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

        # Return probability of class 1 (success)
        prob = self._model.predict_proba(x_scaled)[0, 1]
        return float(prob)
