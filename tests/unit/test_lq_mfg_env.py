"""
Unit tests for LQ-MFG environment.

Tests linear dynamics, quadratic costs, and MFG coupling.
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
    from mfg_pde.alg.reinforcement.environments.lq_mfg_env import LQMFGEnv


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestLQMFGEnv:
    """Tests for LQ-MFG environment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = LQMFGEnv(num_agents=50, x_max=10.0, v_max=5.0, u_max=2.0)

        assert env.num_agents == 50
        assert env.state_dim == 2
        assert env.action_dim == 1
        assert env.x_max == 10.0
        assert env.v_max == 5.0
        assert env.u_max == 2.0

    def test_observation_space(self):
        """Test observation space is correctly configured."""
        env = LQMFGEnv(x_max=10.0, v_max=5.0)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (2,)
        np.testing.assert_array_equal(env.observation_space.low, [-10.0, -5.0])
        np.testing.assert_array_equal(env.observation_space.high, [10.0, 5.0])

    def test_action_space(self):
        """Test action space is correctly configured."""
        env = LQMFGEnv(u_max=2.0)

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        np.testing.assert_array_equal(env.action_space.low, [-2.0])
        np.testing.assert_array_equal(env.action_space.high, [2.0])

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = LQMFGEnv(num_agents=50, x_max=10.0, v_max=5.0)
        state, info = env.reset(seed=42)

        # Check shape
        assert state.shape == (2,)

        # Check within bounds
        assert -10.0 <= state[0] <= 10.0  # position
        assert -5.0 <= state[1] <= 5.0  # velocity

        # Check info
        assert isinstance(info, dict)
        assert "step" in info

    def test_reset_with_seed_reproducible(self):
        """Test seeded reset produces same initial state."""
        env1 = LQMFGEnv(num_agents=50)
        env2 = LQMFGEnv(num_agents=50)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_step_applies_linear_dynamics(self):
        """Test step applies linear position-velocity dynamics."""
        env = LQMFGEnv(num_agents=50, dt=0.1, noise_std=0.0)  # No noise for deterministic test
        state, _ = env.reset(seed=42)

        x0, v0 = state[0], state[1]
        u = np.array([1.0], dtype=np.float32)  # Positive acceleration

        next_state, _, _, _, _ = env.step(u)
        x1, v1 = next_state[0], next_state[1]

        # Check linear dynamics (approximately, with small dt)
        # x' = x + v*dt
        # v' = v + u*dt (with noise_std=0)
        assert x1 > x0  # Position should increase if v0 > 0
        assert v1 > v0  # Velocity should increase with positive control

    def test_reward_penalizes_large_position(self):
        """Test reward penalizes being far from origin."""
        env = LQMFGEnv(cost_state=1.0, cost_control=0.0, cost_mean_field=0.0)

        # State far from origin
        env.reset(seed=42)
        env.agent_states[0] = np.array([5.0, 0.0], dtype=np.float32)

        action = np.array([0.0], dtype=np.float32)
        _, reward_far, _, _, _ = env.step(action)

        # State near origin
        env.reset(seed=42)
        env.agent_states[0] = np.array([1.0, 0.0], dtype=np.float32)

        _, reward_near, _, _, _ = env.step(action)

        # Reward should be better (less negative) when closer to origin
        assert reward_near > reward_far

    def test_reward_penalizes_large_control(self):
        """Test reward penalizes large control actions."""
        env = LQMFGEnv(cost_state=0.0, cost_control=1.0, cost_mean_field=0.0)
        env.reset(seed=42)
        env.agent_states[0] = np.array([0.0, 0.0], dtype=np.float32)

        # Large control
        action_large = np.array([2.0], dtype=np.float32)
        _, reward_large, _, _, _ = env.step(action_large)

        # Small control
        env.agent_states[0] = np.array([0.0, 0.0], dtype=np.float32)
        action_small = np.array([0.5], dtype=np.float32)
        _, reward_small, _, _, _ = env.step(action_small)

        # Smaller control should give better reward
        assert reward_small > reward_large

    def test_mean_field_coupling_computed(self):
        """Test mean field coupling contributes to reward."""
        env = LQMFGEnv(cost_state=0.0, cost_control=0.0, cost_mean_field=1.0)
        state, _ = env.reset(seed=42)

        action = np.array([0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should be non-zero due to mean field term
        assert reward != 0.0
        # Should be negative (cost)
        assert reward < 0.0

    def test_reward_combines_all_terms(self):
        """Test total reward combines state + control + mean field."""
        env = LQMFGEnv(cost_state=1.0, cost_control=0.1, cost_mean_field=0.5)
        env.reset(seed=42)

        # State away from origin with control
        env.agent_states[0] = np.array([2.0, 0.0], dtype=np.float32)
        action = np.array([1.0], dtype=np.float32)

        _, reward, _, _, _ = env.step(action)

        # Reward should be negative (all costs)
        assert reward < 0.0

    def test_population_histogram_computed(self):
        """Test population histogram is computed over position space."""
        env = LQMFGEnv(num_agents=100, population_bins=50)
        env.reset(seed=42)

        histogram = env.get_population_state()

        # Check shape
        assert histogram.shape == (50,)

        # Check normalization
        assert np.isclose(histogram.sum(), 1.0)

        # Check all non-negative
        assert np.all(histogram >= 0)

    def test_episode_runs_to_completion(self):
        """Test episode runs for max_steps without early termination."""
        env = LQMFGEnv(max_steps=10)
        env.reset(seed=42)

        action = np.array([0.0], dtype=np.float32)

        for _ in range(9):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Final step should truncate
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated  # No early termination
        assert truncated  # Time limit reached

    def test_state_clipping_to_bounds(self):
        """Test states are clipped to bounds."""
        env = LQMFGEnv(x_max=10.0, v_max=5.0, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        # Set state near boundary
        env.agent_states[0] = np.array([9.5, 4.5], dtype=np.float32)

        # Large control that would exceed bounds
        action = np.array([5.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Should be clipped to bounds
        assert -10.0 <= next_state[0] <= 10.0
        assert -5.0 <= next_state[1] <= 5.0

    def test_action_clipping_to_bounds(self):
        """Test actions are clipped to bounds."""
        env = LQMFGEnv(u_max=2.0)
        env.reset(seed=42)

        # Action exceeding bounds
        action = np.array([10.0], dtype=np.float32)

        # Should not raise error (clipped internally)
        _, _, _, _, _ = env.step(action)

    def test_custom_cost_weights(self):
        """Test custom cost weights affect reward."""
        # High state cost
        env_high_state = LQMFGEnv(cost_state=10.0, cost_control=0.1, cost_mean_field=0.1)
        env_high_state.reset(seed=42)
        env_high_state.agent_states[0] = np.array([2.0, 0.0], dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)
        _, reward_high_state, _, _, _ = env_high_state.step(action)

        # Low state cost
        env_low_state = LQMFGEnv(cost_state=0.1, cost_control=0.1, cost_mean_field=0.1)
        env_low_state.reset(seed=42)
        env_low_state.agent_states[0] = np.array([2.0, 0.0], dtype=np.float32)
        _, reward_low_state, _, _, _ = env_low_state.step(action)

        # High state cost should give worse reward
        assert reward_high_state < reward_low_state

    def test_agent_positions_initialized_near_origin(self):
        """Test initial agent positions are near origin."""
        env = LQMFGEnv(num_agents=100, x_max=10.0)
        env.reset(seed=42)

        positions = env.agent_states[:, 0]

        # Most agents should be within ±3 (roughly 1 std of Gaussian)
        within_one_std = np.sum(np.abs(positions) <= 3.0)
        assert within_one_std >= 60  # At least 60% within 1 std (conservative)

    def test_render_does_not_crash(self):
        """Test render method exists and doesn't crash."""
        env = LQMFGEnv()
        env.reset(seed=42)
        env.render()  # Should not raise

    def test_close_does_not_crash(self):
        """Test close method exists and doesn't crash."""
        env = LQMFGEnv()
        env.reset(seed=42)
        env.close()  # Should not raise

    def test_drift_is_linear_in_velocity(self):
        """Test drift term is linear in velocity."""
        env = LQMFGEnv()
        env.reset(seed=42)

        state1 = np.array([0.0, 1.0], dtype=np.float32)  # v = 1
        state2 = np.array([0.0, 2.0], dtype=np.float32)  # v = 2
        action = np.array([0.0], dtype=np.float32)
        population = env.get_population_state()

        drift1 = env._drift(state1, action, population)
        drift2 = env._drift(state2, action, population)

        # Position drift should be proportional to velocity
        assert drift2[0] == 2.0 * drift1[0]

    def test_drift_is_linear_in_control(self):
        """Test drift term is linear in control."""
        env = LQMFGEnv()
        env.reset(seed=42)

        state = np.array([0.0, 0.0], dtype=np.float32)
        action1 = np.array([1.0], dtype=np.float32)
        action2 = np.array([2.0], dtype=np.float32)
        population = env.get_population_state()

        drift1 = env._drift(state, action1, population)
        drift2 = env._drift(state, action2, population)

        # Velocity drift should be proportional to control
        assert drift2[1] == 2.0 * drift1[1]
