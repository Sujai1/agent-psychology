"""Uncertainty-aware difficulty predictor using Bayesian Ridge regression.

Instead of predicting a point estimate b_hat and computing
P(success) = sigmoid(theta - b_hat), this approach uses BayesianRidge to
obtain a posterior distribution over b and marginalizes:

    P(success) = E_b[sigmoid(theta - b)]

where b ~ N(mu, sigma^2) from the posterior.

This produces better-calibrated probabilities in the frontier region because
uncertainty in b gets "averaged out" -- when the model is uncertain about
difficulty, predicted probabilities are pulled toward 0.5 rather than being
overconfident.

Two marginalization strategies are provided:
1. Analytic (default): probit approximation
       sigmoid((theta - mu) / sqrt(1 + pi * sigma^2 / 8))
2. Sampling: Monte Carlo average over posterior samples
       (1/N) * sum(sigmoid(theta - b_k)) for b_k ~ N(mu, sigma^2)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.feature_source import TaskFeatureSource


class UncertaintyPredictor:
    """CVPredictor that marginalizes over posterior uncertainty in task difficulty.

    Uses BayesianRidge to fit a distribution over difficulty b for each task,
    then integrates out that uncertainty when computing P(success).

    Implements the CVPredictor protocol:
        fit(data, train_task_ids) -> None
        predict_probability(data, agent_id, task_id) -> float
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        n_samples: int = 100,
        use_analytic: bool = True,
    ) -> None:
        """Initialize the uncertainty predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            n_samples: Number of posterior samples for Monte Carlo
                marginalization. Only used when use_analytic=False.
            use_analytic: If True (default), use the probit approximation
                for marginalization (faster, no sampling noise). If False,
                use Monte Carlo sampling with n_samples draws.
        """
        self._source = source
        self._n_samples = n_samples
        self._use_analytic = use_analytic

        # Model state (populated by fit)
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[BayesianRidge] = None

        # Cached posterior parameters for test tasks: task_id -> (mean_b, std_b)
        self._posterior_cache: Dict[str, Tuple[float, float]] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit BayesianRidge on training task features and difficulties.

        Args:
            data: ExperimentData containing IRT parameters and responses.
            train_task_ids: Task IDs to train on.

        Raises:
            ValueError: If any train task is missing from the feature source.
        """
        # Get features and ground truth difficulties
        X = self._source.get_features(train_task_ids)
        y = data.train_items.loc[train_task_ids, "b"].values

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit BayesianRidge with default priors
        self._model = BayesianRidge()
        self._model.fit(X_scaled, y)

        # Clear cached predictions from previous fold
        self._posterior_cache = {}

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict P(success) by marginalizing over posterior difficulty.

        On first call, computes posterior (mean, std) for all test tasks and
        caches the result.

        Args:
            data: ExperimentData for accessing agent abilities.
            agent_id: The agent whose ability theta is used.
            task_id: The task whose difficulty posterior is marginalized over.

        Returns:
            Predicted probability of success in [0, 1].

        Raises:
            RuntimeError: If called before fit().
            ValueError: If agent_id is not found in abilities.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily compute posterior parameters for all test tasks
        if not self._posterior_cache:
            self._compute_posteriors(data.test_tasks)

        if task_id not in self._posterior_cache:
            raise ValueError(
                f"Task {task_id} not found in posterior cache. "
                f"It may not be in the test set."
            )

        if agent_id not in data.train_abilities.index:
            raise ValueError(f"Agent {agent_id} not found in train_abilities")

        theta = float(data.train_abilities.loc[agent_id, "ability"])
        mean_b, std_b = self._posterior_cache[task_id]

        if self._use_analytic:
            return self._predict_analytic(theta, mean_b, std_b)
        else:
            return self._predict_sampling(theta, mean_b, std_b)

    def _compute_posteriors(self, task_ids: List[str]) -> None:
        """Compute posterior (mean, std) for a batch of tasks.

        Args:
            task_ids: Task IDs to compute posteriors for.
        """
        X = self._source.get_features(task_ids)
        X_scaled = self._scaler.transform(X)

        means, stds = self._model.predict(X_scaled, return_std=True)

        for task_id, mean_b, std_b in zip(task_ids, means, stds):
            self._posterior_cache[task_id] = (float(mean_b), float(std_b))

    @staticmethod
    def _predict_analytic(theta: float, mean_b: float, std_b: float) -> float:
        """Probit approximation to E_b[sigmoid(theta - b)] where b ~ N(mean_b, std_b^2).

        Uses the identity:
            E[sigmoid(x)] approx sigmoid(x / sqrt(1 + pi * var / 8))
        where x ~ N(mu_x, var) and mu_x = theta - mean_b, var = std_b^2.

        This approximation is tight because sigmoid is well-approximated by
        the probit (Gaussian CDF) function, for which the integral is exact.

        Args:
            theta: Agent ability parameter.
            mean_b: Posterior mean of task difficulty.
            std_b: Posterior standard deviation of task difficulty.

        Returns:
            Marginalized probability of success.
        """
        scale = np.sqrt(1.0 + np.pi * std_b**2 / 8.0)
        return float(sigmoid((theta - mean_b) / scale))

    def _predict_sampling(self, theta: float, mean_b: float, std_b: float) -> float:
        """Monte Carlo estimate of E_b[sigmoid(theta - b)] where b ~ N(mean_b, std_b^2).

        Draws n_samples from the posterior and averages sigmoid(theta - b_k).

        Args:
            theta: Agent ability parameter.
            mean_b: Posterior mean of task difficulty.
            std_b: Posterior standard deviation of task difficulty.

        Returns:
            Marginalized probability of success.
        """
        rng = np.random.default_rng()
        b_samples = rng.normal(loc=mean_b, scale=std_b, size=self._n_samples)
        probs = sigmoid(theta - b_samples)
        return float(np.mean(probs))
