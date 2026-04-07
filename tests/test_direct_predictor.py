"""Tests for DirectPredictor."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from experiment_new_tasks.dataset import ExperimentData
from experiment_new_tasks.direct_predictor import DirectPredictor
from experiment_new_tasks.feature_source import TaskFeatureSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeFeatureSource(TaskFeatureSource):
    """In-memory feature source for testing."""

    def __init__(self, features: Dict[str, np.ndarray]) -> None:
        self._features = features
        dims = {v.shape[0] for v in features.values()}
        assert len(dims) == 1, "All feature vectors must have the same dimension"
        self._dim = dims.pop()

    @property
    def name(self) -> str:
        return "FakeFeatures"

    @property
    def task_ids(self) -> List[str]:
        return list(self._features.keys())

    @property
    def feature_dim(self) -> int:
        return self._dim

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        missing = [t for t in task_ids if t not in self._features]
        if missing:
            raise ValueError(f"Missing tasks: {missing}")
        return np.vstack([self._features[t] for t in task_ids])


def _make_experiment_data(
    agent_abilities: Dict[str, float],
    task_difficulties: Dict[str, float],
    responses: Dict[str, Dict[str, int]],
    train_tasks: List[str],
    test_tasks: List[str],
) -> ExperimentData:
    """Build a minimal ExperimentData for testing."""
    all_tasks = train_tasks + test_tasks

    abilities_df = pd.DataFrame(
        {"ability": agent_abilities},
    )
    abilities_df.index.name = "agent_id"

    items_df = pd.DataFrame(
        {"b": task_difficulties},
    )
    items_df.index.name = "task_id"

    return ExperimentData(
        responses=responses,
        train_abilities=abilities_df,
        train_items=items_df,
        full_abilities=abilities_df,
        full_items=items_df,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scenario():
    """Create a synthetic scenario where easy tasks (low feature value) are
    solved by most agents and hard tasks (high feature value) are solved
    only by strong agents.

    Returns (source, data) tuple.
    """
    rng = np.random.RandomState(42)

    n_train_tasks = 60
    n_test_tasks = 20
    n_agents = 30

    train_tasks = [f"train_{i}" for i in range(n_train_tasks)]
    test_tasks = [f"test_{i}" for i in range(n_test_tasks)]
    all_tasks = train_tasks + test_tasks

    # Single feature: "hardness" in [0, 1]
    task_hardness = {t: rng.uniform(0, 1) for t in all_tasks}
    features = {t: np.array([h], dtype=np.float32) for t, h in task_hardness.items()}

    # Agent abilities linearly spaced
    agents = [f"agent_{i}" for i in range(n_agents)]
    agent_abilities = {a: float(v) for a, v in zip(agents, np.linspace(-2, 2, n_agents))}

    # Generate responses: P(success) depends on ability - hardness*4
    # (hardness scaled so range is comparable to ability)
    task_difficulties = {}
    responses: Dict[str, Dict[str, int]] = {}
    for agent_id in agents:
        responses[agent_id] = {}
        theta = agent_abilities[agent_id]
        for task_id in all_tasks:
            h = task_hardness[task_id]
            difficulty = h * 4 - 2  # map [0,1] -> [-2,2]
            task_difficulties[task_id] = difficulty
            logit = theta - difficulty
            prob = 1 / (1 + np.exp(-logit))
            responses[agent_id][task_id] = int(rng.uniform() < prob)

    source = FakeFeatureSource(features)
    data = _make_experiment_data(
        agent_abilities=agent_abilities,
        task_difficulties=task_difficulties,
        responses=responses,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
    )
    return source, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDirectPredictor:
    def test_fit_and_predict_runs(self, synthetic_scenario):
        """Basic smoke test: fit + predict_probability runs without error."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        prob = predictor.predict_probability(data, "agent_0", "test_0")
        assert 0.0 <= prob <= 1.0

    def test_predict_before_fit_raises(self, synthetic_scenario):
        """Calling predict_probability before fit should raise."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)

        with pytest.raises(RuntimeError, match="must be fit"):
            predictor.predict_probability(data, "agent_0", "test_0")

    def test_probabilities_in_valid_range(self, synthetic_scenario):
        """All predicted probabilities should be in [0, 1]."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        for agent_id in data.get_all_agents()[:5]:
            for task_id in data.test_tasks[:5]:
                prob = predictor.predict_probability(data, agent_id, task_id)
                assert 0.0 <= prob <= 1.0, f"prob={prob} for ({agent_id}, {task_id})"

    def test_stronger_agents_have_higher_probs(self, synthetic_scenario):
        """On average, agents with higher ability should get higher predicted
        probabilities across the same set of test tasks."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        agents = data.get_all_agents()
        # Pick weakest and strongest agents
        weakest = agents[0]  # agent_0, ability = -2
        strongest = agents[-1]  # agent_29, ability = 2

        weak_probs = [
            predictor.predict_probability(data, weakest, t) for t in data.test_tasks
        ]
        strong_probs = [
            predictor.predict_probability(data, strongest, t) for t in data.test_tasks
        ]

        assert np.mean(strong_probs) > np.mean(weak_probs), (
            f"Strongest agent mean prob ({np.mean(strong_probs):.3f}) should exceed "
            f"weakest agent mean prob ({np.mean(weak_probs):.3f})"
        )

    def test_missing_agent_raises(self, synthetic_scenario):
        """Predicting for an unknown agent should raise ValueError."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        with pytest.raises(ValueError, match="not found in train_abilities"):
            predictor.predict_probability(data, "nonexistent_agent", "test_0")

    def test_missing_task_raises(self, synthetic_scenario):
        """Predicting for an unknown task should raise ValueError."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        with pytest.raises(ValueError, match="not found in cached test features"):
            predictor.predict_probability(data, "agent_0", "nonexistent_task")

    def test_caches_cleared_between_folds(self, synthetic_scenario):
        """Fitting again should clear cached test features."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)

        # First fit + predict
        predictor.fit(data, data.train_tasks)
        prob1 = predictor.predict_probability(data, "agent_0", "test_0")

        # Second fit (simulating a new fold) should clear cache
        predictor.fit(data, data.train_tasks)
        assert predictor._cached_test_features is None

        # predict_probability should still work after re-fit
        prob2 = predictor.predict_probability(data, "agent_0", "test_0")
        assert 0.0 <= prob2 <= 1.0

    def test_reasonable_accuracy_on_synthetic_data(self, synthetic_scenario):
        """The model should achieve reasonable accuracy on synthetic data
        where the true generative process is simple."""
        source, data = synthetic_scenario
        predictor = DirectPredictor(source)
        predictor.fit(data, data.train_tasks)

        correct = 0
        total = 0
        for agent_id in data.get_all_agents():
            for task_id in data.test_tasks:
                prob = predictor.predict_probability(data, agent_id, task_id)
                actual = data.responses[agent_id][task_id]
                predicted = 1 if prob >= 0.5 else 0
                if predicted == actual:
                    correct += 1
                total += 1

        accuracy = correct / total
        # With a clean synthetic signal, GBM should get well above chance (50%)
        assert accuracy > 0.60, f"Accuracy {accuracy:.3f} is too low on synthetic data"
