"""Tests for WeightedRidgePredictor."""

import numpy as np
import pandas as pd
import pytest

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.weighted_ridge_predictor import WeightedRidgePredictor
from experiment_new_tasks.feature_source import TaskFeatureSource
from typing import Dict, List, Optional


class DummyFeatureSource(TaskFeatureSource):
    """Simple in-memory feature source for testing."""

    def __init__(self, features: Dict[str, np.ndarray]):
        self._features = features
        self._task_ids = list(features.keys())
        self._dim = next(iter(features.values())).shape[0]

    @property
    def name(self) -> str:
        return "DummyFeatures"

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids.copy()

    @property
    def feature_dim(self) -> int:
        return self._dim

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        missing = [t for t in task_ids if t not in self._features]
        if missing:
            raise ValueError(f"Missing tasks: {missing}")
        return np.array([self._features[t] for t in task_ids], dtype=np.float32)


def _make_experiment_data(
    n_agents: int = 10,
    n_train_tasks: int = 20,
    n_test_tasks: int = 5,
    seed: int = 42,
) -> tuple:
    """Build synthetic ExperimentData and a DummyFeatureSource.

    Returns (data, source).
    """
    rng = np.random.RandomState(seed)

    train_tasks = [f"task_train_{i}" for i in range(n_train_tasks)]
    test_tasks = [f"task_test_{i}" for i in range(n_test_tasks)]
    all_tasks = train_tasks + test_tasks
    agent_ids = [f"agent_{i}" for i in range(n_agents)]

    # True abilities and difficulties
    abilities = rng.randn(n_agents)
    difficulties = rng.randn(n_train_tasks + n_test_tasks) * 2

    # Build response matrix from IRT model
    responses: Dict[str, Dict[str, int]] = {}
    for i, agent_id in enumerate(agent_ids):
        agent_resp = {}
        for j, task_id in enumerate(all_tasks):
            p = 1.0 / (1.0 + np.exp(-(abilities[i] - difficulties[j])))
            agent_resp[task_id] = int(rng.rand() < p)
        responses[agent_id] = agent_resp

    # Build DataFrames
    ability_df = pd.DataFrame(
        {"ability": abilities}, index=agent_ids
    )
    items_df = pd.DataFrame(
        {"b": difficulties}, index=all_tasks
    )

    data = ExperimentData(
        responses=responses,
        train_abilities=ability_df,
        train_items=items_df,
        full_abilities=ability_df,
        full_items=items_df,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
    )

    # Build feature source: use difficulty + noise as features so ridge can learn
    feature_dim = 5
    features = {}
    for j, task_id in enumerate(all_tasks):
        feat = rng.randn(feature_dim).astype(np.float32)
        feat[0] = difficulties[j] + rng.randn() * 0.1  # first feature ~ difficulty
        features[task_id] = feat

    source = DummyFeatureSource(features)
    return data, source


class TestFrontierWeights:
    """Test that frontier weight computation is correct."""

    def test_weight_at_half(self):
        """Tasks with 50% solve rate should get weight=1.0."""
        predictor = WeightedRidgePredictor(DummyFeatureSource({"t": np.zeros(1)}))
        responses = {
            "a1": {"t": 1},
            "a2": {"t": 0},
        }
        weights = predictor._compute_frontier_weights(["t"], responses)
        assert np.isclose(weights[0], 1.0), f"Expected 1.0, got {weights[0]}"

    def test_weight_at_extremes(self):
        """Tasks with 0% or 100% solve rate should get weight=0.0."""
        predictor = WeightedRidgePredictor(DummyFeatureSource({"t": np.zeros(1)}))

        # All solve
        responses_all = {"a1": {"t": 1}, "a2": {"t": 1}}
        w_all = predictor._compute_frontier_weights(["t"], responses_all)
        assert np.isclose(w_all[0], 0.0), f"Expected 0.0 for all-solved, got {w_all[0]}"

        # None solve
        responses_none = {"a1": {"t": 0}, "a2": {"t": 0}}
        w_none = predictor._compute_frontier_weights(["t"], responses_none)
        assert np.isclose(w_none[0], 0.0), f"Expected 0.0 for none-solved, got {w_none[0]}"

    def test_weight_at_10_percent(self):
        """Task with 10% solve rate should get weight=0.36."""
        predictor = WeightedRidgePredictor(DummyFeatureSource({"t": np.zeros(1)}))
        responses = {f"a{i}": {"t": 1 if i == 0 else 0} for i in range(10)}
        weights = predictor._compute_frontier_weights(["t"], responses)
        assert np.isclose(weights[0], 0.36), f"Expected 0.36, got {weights[0]}"

    def test_weight_symmetry(self):
        """w(p) should equal w(1-p)."""
        predictor = WeightedRidgePredictor(DummyFeatureSource({"t1": np.zeros(1), "t2": np.zeros(1)}))
        # t1: 20% solve rate, t2: 80% solve rate
        responses = {}
        for i in range(10):
            responses[f"a{i}"] = {
                "t1": 1 if i < 2 else 0,
                "t2": 1 if i < 8 else 0,
            }
        weights = predictor._compute_frontier_weights(["t1", "t2"], responses)
        assert np.isclose(weights[0], weights[1]), (
            f"Weights should be symmetric: w(0.2)={weights[0]}, w(0.8)={weights[1]}"
        )

    def test_zero_responses_raises(self):
        """Task with no responses should raise ValueError."""
        predictor = WeightedRidgePredictor(DummyFeatureSource({"t": np.zeros(1)}))
        responses = {"a1": {}}  # no response for "t"
        with pytest.raises(ValueError, match="zero responses"):
            predictor._compute_frontier_weights(["t"], responses)


class TestCVPredictorProtocol:
    """Test that WeightedRidgePredictor satisfies the CVPredictor protocol."""

    def test_fit_and_predict(self):
        """Basic fit + predict_probability roundtrip."""
        data, source = _make_experiment_data()
        predictor = WeightedRidgePredictor(source)

        predictor.fit(data, data.train_tasks)

        # Predict for each test task and agent
        for task_id in data.test_tasks:
            for agent_id in data.get_all_agents():
                prob = predictor.predict_probability(data, agent_id, task_id)
                assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_predict_before_fit_raises(self):
        """Calling predict_probability before fit should raise RuntimeError."""
        data, source = _make_experiment_data()
        predictor = WeightedRidgePredictor(source)

        with pytest.raises(RuntimeError, match="must be fit"):
            predictor.predict_probability(data, "agent_0", "task_test_0")

    def test_probabilities_vary_by_ability(self):
        """Higher-ability agents should have higher predicted probabilities."""
        data, source = _make_experiment_data(n_agents=50, n_train_tasks=40, seed=123)
        predictor = WeightedRidgePredictor(source)
        predictor.fit(data, data.train_tasks)

        # Pick a test task and compare agents with very different abilities
        task_id = data.test_tasks[0]
        abilities = data.train_abilities["ability"]
        low_agent = abilities.idxmin()
        high_agent = abilities.idxmax()

        prob_low = predictor.predict_probability(data, low_agent, task_id)
        prob_high = predictor.predict_probability(data, high_agent, task_id)

        assert prob_high > prob_low, (
            f"Higher-ability agent should have higher probability: "
            f"p({high_agent})={prob_high}, p({low_agent})={prob_low}"
        )

    def test_refit_clears_cache(self):
        """Calling fit again should clear cached predictions."""
        data, source = _make_experiment_data()
        predictor = WeightedRidgePredictor(source)

        predictor.fit(data, data.train_tasks)
        _ = predictor.predict_probability(data, "agent_0", data.test_tasks[0])
        assert len(predictor._predicted_difficulties) > 0

        # Re-fit should clear cache
        predictor.fit(data, data.train_tasks)
        assert len(predictor._predicted_difficulties) == 0

    def test_custom_alphas(self):
        """Custom alpha list should be used."""
        data, source = _make_experiment_data()
        custom_alphas = [1.0, 100.0]
        predictor = WeightedRidgePredictor(source, alphas=custom_alphas)

        predictor.fit(data, data.train_tasks)
        assert predictor._model.alpha_ in custom_alphas
