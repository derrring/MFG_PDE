"""
Unit tests for ContinuousMFGEnvBase.

Tests the base class API, Gymnasium compatibility, and abstract method enforcement.
"""

from __future__ import annotations

import pytest

try:
    import gymnasium as gym

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

import numpy as np

from mfg_pde.alg.reinforcement.environments import CONTINUOUS_MFG_AVAILABLE

if CONTINUOUS_MFG_AVAILABLE:
    from mfg_pde.alg.reinforcement.environments import ContinuousMFGEnvBase


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestContinuousMFGEnvBase:
    """Tests for ContinuousMFGEnvBase."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ContinuousMFGEnvBase(num_agents=10, state_dim=2, action_dim=1, action_bounds=(-1, 1), population_bins=50)

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be instantiated."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            """Simple concrete implementation for testing."""

            def _get_state_bounds(self):
                low = np.array([-10.0, -5.0], dtype=np.float32)
                high = np.array([10.0, 5.0], dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return action.copy()

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, action_bounds=(-1, 1), population_bins=50)

        assert env.num_agents == 10
        assert env.state_dim == 2
        assert env.action_dim == 1
        assert env.population_bins == 50

    def test_gymnasium_spaces_initialized(self):
        """Test that Gymnasium observation and action spaces are properly initialized."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.array([-10.0, -5.0], dtype=np.float32)
                high = np.array([10.0, 5.0], dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return action

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, action_bounds=(-2, 2))

        # Check observation space
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (2,)
        np.testing.assert_array_equal(env.observation_space.low, [-10.0, -5.0])
        np.testing.assert_array_equal(env.observation_space.high, [10.0, 5.0])

        # Check action space
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        np.testing.assert_array_equal(env.action_space.low, [-2.0])
        np.testing.assert_array_equal(env.action_space.high, [2.0])

    def test_reset_initializes_environment(self):
        """Test reset() initializes environment correctly."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.random.randn(self.num_agents, self.state_dim).astype(np.float32)

            def _drift(self, state, action, population):
                return action

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1)
        state, info = env.reset(seed=42)

        assert state.shape == (2,)
        assert isinstance(info, dict)
        assert "step" in info
        assert info["step"] == 0
        assert env.current_step == 0
        assert env.agent_states is not None
        assert env.agent_states.shape == (10, 2)

    def test_reset_with_seed_reproducible(self):
        """Test reset() with seed produces reproducible results."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return self.rng.standard_normal((self.num_agents, self.state_dim)).astype(np.float32)

            def _drift(self, state, action, population):
                return action

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env1 = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1)
        env2 = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_step_returns_correct_tuple(self):
        """Test step() returns (state, reward, terminated, truncated, info) tuple."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return action

            def compute_mean_field_coupling(self, state, population):
                return -0.1

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, max_steps=10)
        env.reset(seed=42)

        action = np.array([0.5], dtype=np.float32)
        result = env.step(action)

        assert len(result) == 5
        state, reward, terminated, truncated, info = result

        assert state.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_applies_dynamics(self):
        """Test step() applies dynamics correctly."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                # Simple dynamics: x' = x + action
                drift = np.zeros(self.state_dim, dtype=np.float32)
                drift[0] = action[0]
                return drift

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, dt=0.1, noise_std=0.0)
        env.reset(seed=42)

        initial_state = env.agent_states[0].copy()
        action = np.array([1.0], dtype=np.float32)
        next_state, _, _, _, _ = env.step(action)

        # Check state changed (approximately by action * dt)
        assert not np.allclose(next_state, initial_state)
        assert next_state[0] > initial_state[0]

    def test_action_clipping(self):
        """Test actions are clipped to bounds."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                # Record received action
                self.last_action = action
                return action

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, action_bounds=(-1.0, 1.0))
        env.reset(seed=42)

        # Action exceeds bounds
        action = np.array([5.0], dtype=np.float32)
        env.step(action)

        # Should be clipped to [−1, 1]
        assert np.all(env.last_action >= -1.0 - 1e-6)
        assert np.all(env.last_action <= 1.0 + 1e-6)

    def test_state_clipping_to_bounds(self):
        """Test states are clipped to state space bounds."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.array([-1.0, -1.0], dtype=np.float32)
                high = np.array([1.0, 1.0], dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.array([[0.9, 0.9]] * self.num_agents, dtype=np.float32)

            def _drift(self, state, action, population):
                # Large drift that would exceed bounds
                return np.array([10.0, 10.0], dtype=np.float32)

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        action = np.array([0.0], dtype=np.float32)
        next_state, _, _, _, _ = env.step(action)

        # State should be clipped to bounds
        assert np.all(next_state >= -1.0 - 1e-5)
        assert np.all(next_state <= 1.0 + 1e-5)

    def test_truncation_at_max_steps(self):
        """Test episode truncates at max_steps."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return np.zeros(self.state_dim, dtype=np.float32)

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, max_steps=5)
        env.reset(seed=42)

        action = np.array([0.0], dtype=np.float32)

        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Step 5 (index 4) should truncate
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert truncated

    def test_reward_combines_individual_and_mean_field(self):
        """Test reward combines individual and mean field components."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return np.zeros(self.state_dim, dtype=np.float32)

            def _individual_reward(self, state, action, next_state):
                return -np.sum(action**2)

            def compute_mean_field_coupling(self, state, population):
                return -0.5

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1)
        env.reset(seed=42)

        action = np.array([1.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should be −1.0 (action cost) − 0.5 (MF coupling) = −1.5
        assert np.isclose(reward, -1.5)

    def test_population_histogram_computed(self):
        """Test population histogram is computed."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                low = np.full(2, -10.0, dtype=np.float32)
                high = np.full(2, 10.0, dtype=np.float32)
                return low, high

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim), dtype=np.float32)

            def _drift(self, state, action, population):
                return np.zeros(self.state_dim, dtype=np.float32)

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        env = SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, population_bins=50)
        env.reset(seed=42)

        population = env.get_population_state()

        assert population.shape == (50,)
        assert np.isclose(np.sum(population), 1.0)

    def test_parameter_validation(self):
        """Test constructor validates parameters."""

        class SimpleMFGEnv(ContinuousMFGEnvBase):
            def _get_state_bounds(self):
                return np.array([-10.0]), np.array([10.0])

            def _sample_initial_states(self):
                return np.zeros((self.num_agents, self.state_dim))

            def _drift(self, state, action, population):
                return np.zeros(self.state_dim)

            def compute_mean_field_coupling(self, state, population):
                return 0.0

        with pytest.raises(ValueError, match="num_agents must be >= 1"):
            SimpleMFGEnv(num_agents=0, state_dim=2, action_dim=1)

        with pytest.raises(ValueError, match="state_dim must be >= 1"):
            SimpleMFGEnv(num_agents=10, state_dim=0, action_dim=1)

        with pytest.raises(ValueError, match="action_dim must be >= 1"):
            SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=0)

        with pytest.raises(ValueError, match="population_bins must be >= 1"):
            SimpleMFGEnv(num_agents=10, state_dim=2, action_dim=1, population_bins=0)
