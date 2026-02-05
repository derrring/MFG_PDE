"""
Unit tests for Multi-Population MFG Environment Base Class.

Tests the abstract base class API and core functionality for multi-population
Mean Field Game environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mfg_pde.alg.reinforcement.environments.multi_population_env import (
    MultiPopulationMFGEnv,
)


class SimpleMultiPopEnv(MultiPopulationMFGEnv):
    """Simple concrete implementation for testing."""

    def _sample_initial_state(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """Sample initial state uniformly in [0, 1]^state_dim."""
        return np.random.uniform(0, 1, size=self.state_dims[pop_id])

    def _compute_population_distribution(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """Compute uniform distribution over population bins."""
        dist = np.ones(self.population_sizes[pop_id]) / self.population_sizes[pop_id]
        return dist

    def _dynamics(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> NDArray[np.floating[Any]]:
        """Simple dynamics: s' = s + dt * action."""
        action_array = np.atleast_1d(action)
        next_state = state + self.dt * action_array[: self.state_dims[pop_id]]
        return np.clip(next_state, 0, 1)  # Keep in [0, 1]

    def _reward(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        next_state: NDArray[np.floating[Any]],
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> float:
        """Simple reward: -||action||^2 (penalize large actions)."""
        action_array = np.atleast_1d(action)
        return -float(np.sum(action_array**2))


class TestMultiPopulationMFGEnv:
    """Test suite for MultiPopulationMFGEnv base class."""

    def test_initialization_two_populations(self):
        """Test initialization with 2 populations."""
        env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 1, "bounds": (0, 1)},
            ],
            population_sizes=50,
        )

        assert env.num_populations == 2
        assert env.state_dims == [2, 2]
        assert env.population_sizes == [50, 50]
        assert len(env.action_specs) == 2

    def test_initialization_three_populations_heterogeneous(self):
        """Test initialization with 3 heterogeneous populations."""
        env = SimpleMultiPopEnv(
            num_populations=3,
            state_dims=[2, 3, 1],
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 3, "bounds": (-2, 2)},
                {"type": "continuous", "dim": 1, "bounds": (0, 1)},
            ],
            population_sizes=[50, 100, 30],
        )

        assert env.num_populations == 3
        assert env.state_dims == [2, 3, 1]
        assert env.population_sizes == [50, 100, 30]

    def test_initialization_fails_single_population(self):
        """Test that initialization fails with N < 2."""
        with pytest.raises(ValueError, match="Multi-population requires N ≥ 2"):
            SimpleMultiPopEnv(
                num_populations=1,
                state_dims=2,
                action_specs=[{"type": "continuous", "dim": 2, "bounds": (-1, 1)}],
                population_sizes=50,
            )

    def test_reset(self):
        """Test environment reset."""
        env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
            ],
            population_sizes=50,
        )

        states, info = env.reset(seed=42)

        # Check states
        assert len(states) == 2
        assert all(pop_id in states for pop_id in range(2))
        assert all(states[i].shape == (2,) for i in range(2))

        # Check info
        assert len(info) == 2
        assert all(pop_id in info for pop_id in range(2))

    def test_step_returns_correct_structure(self):
        """Test that step returns correct dictionary structure."""
        env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
            ],
            population_sizes=50,
        )

        env.reset(seed=42)

        actions = {0: np.array([0.5, 0.5]), 1: np.array([-0.3, 0.2])}

        next_states, rewards, terminated, truncated, info = env.step(actions)

        # Check all returns are dictionaries
        assert isinstance(next_states, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)

        # Check all have correct keys
        assert set(next_states.keys()) == {0, 1}
        assert set(rewards.keys()) == {0, 1}

    def test_get_population_states(self):
        """Test get_population_states method."""
        env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
            ],
            population_sizes=50,
        )

        env.reset(seed=42)
        pop_states = env.get_population_states()

        assert len(pop_states) == 2
        assert pop_states[0].shape == (50,)
        assert pop_states[1].shape == (50,)

        # Check normalization
        assert np.isclose(np.sum(pop_states[0]), 1.0)
        assert np.isclose(np.sum(pop_states[1]), 1.0)
