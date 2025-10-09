"""
Unit tests for multi-population environment.

Tests MultiPopulationMFGEnvironment base class and SimpleMultiPopulationEnv.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import pytest

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.multi_population.base_environment import (
        SimpleMultiPopulationEnv,
    )
if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.multi_population.population_config import (
        PopulationConfig,
    )


@pytest.fixture
def simple_populations():
    """Create simple population configurations for testing."""
    return {
        "pop1": PopulationConfig(
            population_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            algorithm="ddpg",
            coupling_weights={"pop2": 0.5},
            initial_distribution=lambda: np.array([0.0, 0.0]),
        ),
        "pop2": PopulationConfig(
            population_id="pop2",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            algorithm="td3",
            coupling_weights={"pop1": 0.3},
            initial_distribution=lambda: np.array([1.0, 1.0]),
        ),
    }


class TestSimpleMultiPopulationEnv:
    """Tests for SimpleMultiPopulationEnv."""

    def test_initialization(self, simple_populations):
        """Test environment initialization."""
        env = SimpleMultiPopulationEnv(
            populations=simple_populations,
            time_horizon=1.0,
            dt=0.1,
        )

        assert env.num_populations == 2
        assert env.population_ids == ["pop1", "pop2"]
        assert env.max_steps == 10

    def test_reset(self, simple_populations):
        """Test environment reset."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        states, info = env.reset()

        assert set(states.keys()) == {"pop1", "pop2"}
        assert np.allclose(states["pop1"], [0.0, 0.0])
        assert np.allclose(states["pop2"], [1.0, 1.0])
        assert "pop1" in info
        assert "pop2" in info

    def test_reset_with_seed(self, simple_populations):
        """Test deterministic reset with seed."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        states1, _ = env.reset(seed=42)
        states2, _ = env.reset(seed=42)

        assert np.allclose(states1["pop1"], states2["pop1"])
        assert np.allclose(states1["pop2"], states2["pop2"])

    def test_step(self, simple_populations):
        """Test environment step."""
        env = SimpleMultiPopulationEnv(populations=simple_populations, dt=0.1)

        states, _ = env.reset()
        actions = {"pop1": np.array([0.5]), "pop2": np.array([-0.5])}

        next_states, rewards, terminated, truncated, _info = env.step(actions)

        # Check state evolution: s' = s + a * dt
        assert np.allclose(next_states["pop1"], states["pop1"] + actions["pop1"] * 0.1)
        assert np.allclose(next_states["pop2"], states["pop2"] + actions["pop2"] * 0.1)

        # Check rewards (should be negative for action cost)
        assert rewards["pop1"] < 0
        assert rewards["pop2"] < 0

        # Check termination
        assert not terminated["pop1"]
        assert not terminated["pop2"]
        assert not truncated["pop1"]
        assert not truncated["pop2"]

    def test_step_invalid_actions(self, simple_populations):
        """Test that invalid actions raise ValueError."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        env.reset()

        # Missing action for pop2
        with pytest.raises(ValueError, match="don't match populations"):
            env.step({"pop1": np.array([0.5])})

    def test_truncation(self, simple_populations):
        """Test episode truncation at max steps."""
        env = SimpleMultiPopulationEnv(
            populations=simple_populations,
            time_horizon=1.0,
            dt=0.1,
        )

        env.reset()
        actions = {"pop1": np.array([0.0]), "pop2": np.array([0.0])}

        # Run for max_steps
        for _ in range(9):
            _, _, _terminated, truncated, _ = env.step(actions)
            assert not truncated["pop1"]

        # Last step should truncate
        _, _, _terminated, truncated, _ = env.step(actions)
        assert truncated["pop1"]
        assert truncated["pop2"]

    def test_get_population_state(self, simple_populations):
        """Test getting individual population state."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        env.reset()
        pop_state = env.get_population_state("pop1")

        assert isinstance(pop_state, np.ndarray)
        assert pop_state.shape == (2,)  # Same as state_dim

    def test_get_all_population_states(self, simple_populations):
        """Test getting all population states."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        env.reset()
        all_states = env.get_all_population_states()

        assert set(all_states.keys()) == {"pop1", "pop2"}
        assert all(isinstance(s, np.ndarray) for s in all_states.values())

    def test_coupling_in_reward(self, simple_populations):
        """Test that coupling weights affect rewards."""
        env = SimpleMultiPopulationEnv(populations=simple_populations, dt=0.1)

        _states, _ = env.reset()

        # Pop1 and pop2 start at different positions
        # Pop1 has coupling_weight=0.5 to pop2
        actions = {"pop1": np.array([0.0]), "pop2": np.array([0.0])}

        _, rewards, _, _, _ = env.step(actions)

        # Reward should include coupling cost
        # Since pop1 couples to pop2 with weight 0.5
        assert rewards["pop1"] < 0  # Has coupling cost


class TestMultiPopulationEnvironmentValidation:
    """Tests for environment validation."""

    def test_too_few_populations(self):
        """Test that single population raises error."""
        populations = {
            "pop1": PopulationConfig(
                population_id="pop1",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            ),
        }

        with pytest.raises(ValueError, match="requires at least 2 populations"):
            SimpleMultiPopulationEnv(populations=populations)

    def test_too_many_populations(self):
        """Test that >5 populations raises error."""
        populations = {
            f"pop_{i}": PopulationConfig(
                population_id=f"pop_{i}",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            )
            for i in range(6)
        }

        with pytest.raises(ValueError, match="supports up to 5 populations"):
            SimpleMultiPopulationEnv(populations=populations)

    def test_invalid_population_id(self, simple_populations):
        """Test getting state for invalid population ID."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        env.reset()

        with pytest.raises(KeyError, match="Unknown population ID"):
            env.get_population_state("invalid")


class TestMultiPopulationDynamics:
    """Tests for multi-population dynamics."""

    def test_linear_dynamics(self, simple_populations):
        """Test that SimpleEnv implements linear dynamics correctly."""
        env = SimpleMultiPopulationEnv(populations=simple_populations, dt=0.1)

        states, _ = env.reset()
        actions = {"pop1": np.array([1.0]), "pop2": np.array([-1.0])}

        next_states, _, _, _, _ = env.step(actions)

        # s' = s + a * dt (action broadcast to all state dimensions)
        expected_pop1 = states["pop1"] + actions["pop1"] * 0.1
        expected_pop2 = states["pop2"] + actions["pop2"] * 0.1

        assert np.allclose(next_states["pop1"], expected_pop1)
        assert np.allclose(next_states["pop2"], expected_pop2)

    def test_quadratic_cost(self, simple_populations):
        """Test that reward includes quadratic action cost."""
        env = SimpleMultiPopulationEnv(populations=simple_populations)

        env.reset()

        # Large action should have larger cost
        actions_small = {"pop1": np.array([0.1]), "pop2": np.array([0.1])}
        actions_large = {"pop1": np.array([1.0]), "pop2": np.array([1.0])}

        env.reset()
        _, rewards_small, _, _, _ = env.step(actions_small)

        env.reset()
        _, rewards_large, _, _, _ = env.step(actions_large)

        # Larger actions should have more negative rewards
        assert rewards_large["pop1"] < rewards_small["pop1"]
        assert rewards_large["pop2"] < rewards_small["pop2"]
