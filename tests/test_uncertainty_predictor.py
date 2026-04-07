"""Tests for UncertaintyPredictor.

Verifies:
1. Basic fit/predict workflow
2. Analytic vs sampling consistency
3. Uncertainty pulls predictions toward 0.5
4. Protocol conformance (CVPredictor interface)
5. Error handling (predict before fit, missing agents)
"""

import numpy as np
import pandas as pd
import pytest

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.feature_source import TaskFeatureSource
from experiment_new_tasks.uncertainty_predictor import UncertaintyPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyFeatureSource(TaskFeatureSource):
    """In-memory feature source for testing."""

    def __init__(self, task_features: dict[str, np.ndarray]):
        self._task_features = task_features
        self._task_ids = list(task_features.keys())
        self._dim = next(iter(task_features.values())).shape[0]

    @property
    def name(self) -> str:
        return "DummyFeatures"

    @property
    def task_ids(self) -> list[str]:
        return list(self._task_ids)

    @property
    def feature_dim(self) -> int:
        return self._dim

    def get_features(self, task_ids: list[str]) -> np.ndarray:
        missing = [t for t in task_ids if t not in self._task_features]
        if missing:
            raise ValueError(f"Missing tasks: {missing}")
        return np.array([self._task_features[t] for t in task_ids])


def _make_experiment_data(
    n_train: int = 50,
    n_test: int = 10,
    n_agents: int = 5,
    feature_dim: int = 4,
    seed: int = 42,
) -> tuple[ExperimentData, DummyFeatureSource]:
    """Create synthetic ExperimentData and matching feature source.

    Difficulty is a linear function of features plus noise, so BayesianRidge
    should achieve reasonable predictions.
    """
    rng = np.random.default_rng(seed)

    all_task_ids = [f"task_{i}" for i in range(n_train + n_test)]
    train_tasks = all_task_ids[:n_train]
    test_tasks = all_task_ids[n_train:]
    agent_ids = [f"agent_{j}" for j in range(n_agents)]

    # Generate features
    all_features = rng.standard_normal((n_train + n_test, feature_dim))
    task_features = {
        tid: all_features[i] for i, tid in enumerate(all_task_ids)
    }

    # True coefficient: difficulty = X @ w + noise
    w_true = rng.standard_normal(feature_dim)
    difficulties = all_features @ w_true + rng.normal(0, 0.3, size=n_train + n_test)

    # Agent abilities spanning a reasonable range
    abilities = np.linspace(-2, 2, n_agents)

    # Generate binary responses from IRT model
    responses: dict[str, dict[str, int]] = {}
    for j, agent_id in enumerate(agent_ids):
        agent_responses: dict[str, int] = {}
        for i, task_id in enumerate(all_task_ids):
            prob = 1.0 / (1.0 + np.exp(-(abilities[j] - difficulties[i])))
            agent_responses[task_id] = int(rng.random() < prob)
        responses[agent_id] = agent_responses

    # Build DataFrames
    items_df = pd.DataFrame(
        {"b": difficulties}, index=all_task_ids
    )
    abilities_df = pd.DataFrame(
        {"ability": abilities}, index=agent_ids
    )

    data = ExperimentData(
        responses=responses,
        train_abilities=abilities_df,
        train_items=items_df,
        full_abilities=abilities_df,
        full_items=items_df,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
    )

    source = DummyFeatureSource(task_features)
    return data, source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUncertaintyPredictor:
    """Test suite for UncertaintyPredictor."""

    def test_fit_and_predict_analytic(self):
        """Basic workflow: fit on train tasks, predict probabilities on test tasks."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)

        predictor.fit(data, data.train_tasks)

        prob = predictor.predict_probability(data, "agent_0", "task_50")
        assert 0.0 <= prob <= 1.0

    def test_fit_and_predict_sampling(self):
        """Sampling-based marginalization produces valid probabilities."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, n_samples=500, use_analytic=False)

        predictor.fit(data, data.train_tasks)

        prob = predictor.predict_probability(data, "agent_0", "task_50")
        assert 0.0 <= prob <= 1.0

    def test_analytic_and_sampling_agree(self):
        """Analytic and sampling methods should produce similar results."""
        data, source = _make_experiment_data()

        predictor_analytic = UncertaintyPredictor(source, use_analytic=True)
        predictor_sampling = UncertaintyPredictor(source, n_samples=10000, use_analytic=False)

        predictor_analytic.fit(data, data.train_tasks)
        predictor_sampling.fit(data, data.train_tasks)

        diffs = []
        for agent_id in ["agent_0", "agent_2", "agent_4"]:
            for task_id in data.test_tasks:
                p_a = predictor_analytic.predict_probability(data, agent_id, task_id)
                p_s = predictor_sampling.predict_probability(data, agent_id, task_id)
                diffs.append(abs(p_a - p_s))

        # With 10k samples, the MC estimate should be within ~0.02 of analytic
        assert np.mean(diffs) < 0.02, f"Mean diff = {np.mean(diffs):.4f}"

    def test_uncertainty_pulls_toward_half(self):
        """Higher posterior uncertainty should pull predictions toward 0.5.

        Compare the uncertainty predictor (analytic) against a point-estimate
        approach. When std_b > 0, the marginalized probability should be closer
        to 0.5 than the point estimate.
        """
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)
        predictor.fit(data, data.train_tasks)

        from scipy.special import expit as sigmoid

        # Trigger cache population by making one prediction
        predictor.predict_probability(data, data.get_all_agents()[0], data.test_tasks[0])

        closer_to_half_count = 0
        total = 0

        for agent_id in data.get_all_agents():
            theta = float(data.train_abilities.loc[agent_id, "ability"])
            for task_id in data.test_tasks:
                mean_b, std_b = predictor._posterior_cache[task_id]
                if std_b < 1e-6:
                    continue  # Skip tasks with negligible uncertainty

                p_marginalized = predictor.predict_probability(data, agent_id, task_id)
                p_point = float(sigmoid(theta - mean_b))

                dist_margin = abs(p_marginalized - 0.5)
                dist_point = abs(p_point - 0.5)

                if dist_margin <= dist_point:
                    closer_to_half_count += 1
                total += 1

        # Marginalized should be closer to 0.5 in most cases
        assert total > 0, "No tasks had non-trivial uncertainty"
        ratio = closer_to_half_count / total
        assert ratio > 0.9, (
            f"Expected marginalized prob to be closer to 0.5 in >90% of cases, "
            f"got {ratio:.1%} ({closer_to_half_count}/{total})"
        )

    def test_predict_before_fit_raises(self):
        """Calling predict_probability before fit should raise RuntimeError."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source)

        with pytest.raises(RuntimeError, match="must be fit"):
            predictor.predict_probability(data, "agent_0", "task_50")

    def test_missing_agent_raises(self):
        """Requesting a nonexistent agent should raise ValueError."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)
        predictor.fit(data, data.train_tasks)

        with pytest.raises(ValueError, match="not found in train_abilities"):
            predictor.predict_probability(data, "nonexistent_agent", "task_50")

    def test_missing_task_raises(self):
        """Requesting a task not in the test set should raise ValueError."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)
        predictor.fit(data, data.train_tasks)

        # Trigger cache population
        predictor.predict_probability(data, "agent_0", data.test_tasks[0])

        with pytest.raises(ValueError, match="not found in posterior cache"):
            predictor.predict_probability(data, "agent_0", "nonexistent_task")

    def test_higher_ability_higher_probability(self):
        """Agents with higher ability should have higher predicted success probability."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)
        predictor.fit(data, data.train_tasks)

        task_id = data.test_tasks[0]
        probs = []
        for agent_id in sorted(data.get_all_agents()):
            prob = predictor.predict_probability(data, agent_id, task_id)
            probs.append(prob)

        # Abilities are monotonically increasing by construction (linspace)
        # So probabilities should be monotonically increasing
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-10, (
                f"Expected monotonically increasing probs, "
                f"but got probs[{i}]={probs[i]:.4f} > probs[{i+1}]={probs[i+1]:.4f}"
            )

    def test_cache_cleared_on_refit(self):
        """Fitting again should clear the posterior cache."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)

        predictor.fit(data, data.train_tasks)
        predictor.predict_probability(data, "agent_0", data.test_tasks[0])
        assert len(predictor._posterior_cache) > 0

        # Refit should clear cache
        predictor.fit(data, data.train_tasks)
        assert len(predictor._posterior_cache) == 0

    def test_all_test_tasks_predicted_at_once(self):
        """First predict call should populate cache for all test tasks."""
        data, source = _make_experiment_data()
        predictor = UncertaintyPredictor(source, use_analytic=True)
        predictor.fit(data, data.train_tasks)

        # One call should populate cache for all test tasks
        predictor.predict_probability(data, "agent_0", data.test_tasks[0])
        assert set(predictor._posterior_cache.keys()) == set(data.test_tasks)
