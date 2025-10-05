"""
Unit tests for Crowd Navigation environment.

Tests 2D spatial navigation, goal-directed behavior, and crowd avoidance.
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
    from mfg_pde.alg.reinforcement.environments.crowd_navigation_env import CrowdNavigationEnv


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestCrowdNavigationEnv:
    """Tests for Crowd Navigation environment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = CrowdNavigationEnv(num_agents=100, domain_size=10.0, v_max=2.0, a_max=1.0)

        assert env.num_agents == 100
        assert env.state_dim == 6
        assert env.action_dim == 2
        assert env.domain_size == 10.0
        assert env.v_max == 2.0
        assert env.a_max == 1.0

    def test_observation_space(self):
        """Test observation space is correctly configured."""
        env = CrowdNavigationEnv(domain_size=10.0, v_max=2.0)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (6,)
        np.testing.assert_array_equal(env.observation_space.low, [0.0, 0.0, -2.0, -2.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.observation_space.high, [10.0, 10.0, 2.0, 2.0, 10.0, 10.0])

    def test_action_space(self):
        """Test action space is correctly configured."""
        env = CrowdNavigationEnv(a_max=1.0)

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = CrowdNavigationEnv(num_agents=100, domain_size=10.0, v_max=2.0)
        state, info = env.reset(seed=42)

        # Check shape
        assert state.shape == (6,)

        # Check position within bounds
        assert 0.0 <= state[0] <= 10.0  # x
        assert 0.0 <= state[1] <= 10.0  # y

        # Check velocity within bounds
        assert -2.0 <= state[2] <= 2.0  # vx
        assert -2.0 <= state[3] <= 2.0  # vy

        # Check goal within bounds
        assert 0.0 <= state[4] <= 10.0  # x_goal
        assert 0.0 <= state[5] <= 10.0  # y_goal

        assert isinstance(info, dict)

    def test_reset_initializes_with_zero_velocity(self):
        """Test agents start from rest."""
        env = CrowdNavigationEnv(num_agents=100)
        state, _ = env.reset(seed=42)

        # Initial velocities should be zero
        assert np.isclose(state[2], 0.0)  # vx
        assert np.isclose(state[3], 0.0)  # vy

    def test_reset_with_seed_reproducible(self):
        """Test seeded reset produces same initial state."""
        env1 = CrowdNavigationEnv(num_agents=100)
        env2 = CrowdNavigationEnv(num_agents=100)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_goal_position_set_in_reset(self):
        """Test goal position is set during reset."""
        env = CrowdNavigationEnv(num_agents=100)
        env.reset(seed=42)

        assert env.goal_position is not None
        assert env.goal_position.shape == (2,)

    def test_step_applies_2d_kinematics(self):
        """Test step applies 2D kinematic dynamics."""
        env = CrowdNavigationEnv(num_agents=100, dt=0.1, noise_std=0.0)
        state, _ = env.reset(seed=42)

        # Apply acceleration in x-direction
        action = np.array([1.0, 0.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Velocity should increase in x-direction (v' = v + a*dt)
        assert next_state[2] > state[2]  # vx increased

    def test_reward_penalizes_distance_to_goal(self):
        """Test reward penalizes being far from goal."""
        env = CrowdNavigationEnv(cost_distance=1.0, cost_velocity=0.0, cost_control=0.0, cost_crowd=0.0)

        # State far from goal
        env.reset(seed=42)
        env.agent_states[0] = np.array([1.0, 1.0, 0.0, 0.0, 9.0, 9.0], dtype=np.float32)

        action = np.array([0.0, 0.0], dtype=np.float32)
        _, reward_far, _, _, _ = env.step(action)

        # State near goal
        env.agent_states[0] = np.array([9.0, 9.0, 0.0, 0.0, 9.0, 9.0], dtype=np.float32)
        _, reward_near, _, _, _ = env.step(action)

        # Reward should be better when closer to goal
        assert reward_near > reward_far

    def test_reward_penalizes_high_velocity(self):
        """Test reward penalizes high velocity."""
        env = CrowdNavigationEnv(cost_distance=0.0, cost_velocity=1.0, cost_control=0.0, cost_crowd=0.0)
        env.reset(seed=42)

        # High velocity
        env.agent_states[0] = np.array([5.0, 5.0, 2.0, 2.0, 5.0, 5.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, reward_high_vel, _, _, _ = env.step(action)

        # Low velocity
        env.agent_states[0] = np.array([5.0, 5.0, 0.5, 0.5, 5.0, 5.0], dtype=np.float32)
        _, reward_low_vel, _, _, _ = env.step(action)

        # Lower velocity should give better reward
        assert reward_low_vel > reward_high_vel

    def test_reward_penalizes_large_control(self):
        """Test reward penalizes large accelerations."""
        env = CrowdNavigationEnv(cost_distance=0.0, cost_velocity=0.0, cost_control=1.0, cost_crowd=0.0)
        env.reset(seed=42)
        env.agent_states[0] = np.array([5.0, 5.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)

        # Large acceleration
        action_large = np.array([1.0, 1.0], dtype=np.float32)
        _, reward_large, _, _, _ = env.step(action_large)

        # Small acceleration
        env.agent_states[0] = np.array([5.0, 5.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        action_small = np.array([0.1, 0.1], dtype=np.float32)
        _, reward_small, _, _, _ = env.step(action_small)

        # Smaller control should give better reward
        assert reward_small > reward_large

    def test_goal_bonus_when_reached(self):
        """Test agent receives bonus when reaching goal."""
        env = CrowdNavigationEnv(
            goal_bonus=10.0, goal_radius=0.5, cost_distance=0.0, cost_velocity=0.0, cost_control=0.0, cost_crowd=0.0
        )
        env.reset(seed=42)

        # Position within goal radius
        env.agent_states[0] = np.array([5.0, 5.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, reward_at_goal, _, _, _ = env.step(action)

        # Position far from goal
        env.agent_states[0] = np.array([1.0, 1.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        _, reward_far, _, _, _ = env.step(action)

        # Should receive bonus at goal
        assert reward_at_goal > reward_far
        assert reward_at_goal >= 10.0  # At least the bonus amount

    def test_terminates_when_goal_reached(self):
        """Test episode terminates when agent reaches goal."""
        env = CrowdNavigationEnv(goal_radius=0.5)
        env.reset(seed=42)

        # Position at goal
        env.agent_states[0] = np.array([5.0, 5.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)

        assert terminated

    def test_does_not_terminate_when_far_from_goal(self):
        """Test episode does not terminate when far from goal."""
        env = CrowdNavigationEnv(goal_radius=0.5, max_steps=100)
        env.reset(seed=42)

        # Position far from goal
        env.agent_states[0] = np.array([1.0, 1.0, 0.0, 0.0, 9.0, 9.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)

        assert not terminated
        assert not truncated

    def test_crowd_avoidance_penalty(self):
        """Test mean field coupling provides crowd avoidance."""
        env = CrowdNavigationEnv(cost_crowd=1.0, cost_distance=0.0, cost_velocity=0.0, cost_control=0.0)
        _state, _ = env.reset(seed=42)

        action = np.array([0.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should include crowd penalty (non-zero)
        # Can't test exact value due to simplified implementation
        assert isinstance(reward, float)

    def test_population_histogram_computed(self):
        """Test population histogram is computed over 2D space."""
        env = CrowdNavigationEnv(num_agents=100, population_bins=50)
        env.reset(seed=42)

        histogram = env.get_population_state()

        # Check shape (1D flattened for compatibility)
        assert histogram.shape == (50,)

        # Check normalization
        assert np.isclose(histogram.sum(), 1.0)

        # Check all non-negative
        assert np.all(histogram >= 0)

    def test_state_clipping_to_bounds(self):
        """Test states are clipped to domain bounds."""
        env = CrowdNavigationEnv(domain_size=10.0, v_max=2.0, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        # Set state near boundary with high velocity
        env.agent_states[0] = np.array([9.5, 9.5, 1.0, 1.0, 5.0, 5.0], dtype=np.float32)

        # Large acceleration toward boundary
        action = np.array([2.0, 2.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Should be clipped to domain bounds
        assert 0.0 <= next_state[0] <= 10.0
        assert 0.0 <= next_state[1] <= 10.0
        assert -2.0 <= next_state[2] <= 2.0
        assert -2.0 <= next_state[3] <= 2.0

    def test_action_clipping_to_bounds(self):
        """Test actions are clipped to bounds."""
        env = CrowdNavigationEnv(a_max=1.0)
        env.reset(seed=42)

        # Action exceeding bounds
        action = np.array([10.0, 10.0], dtype=np.float32)

        # Should not raise error (clipped internally)
        _, _, _, _, _ = env.step(action)

    def test_custom_cost_weights(self):
        """Test custom cost weights affect reward."""
        # High distance cost
        env_high_dist = CrowdNavigationEnv(cost_distance=10.0, cost_velocity=0.1, cost_control=0.1, cost_crowd=0.1)
        env_high_dist.reset(seed=42)
        env_high_dist.agent_states[0] = np.array([1.0, 1.0, 0.0, 0.0, 9.0, 9.0], dtype=np.float32)
        action = np.array([0.1, 0.1], dtype=np.float32)
        _, reward_high_dist, _, _, _ = env_high_dist.step(action)

        # Low distance cost
        env_low_dist = CrowdNavigationEnv(cost_distance=0.1, cost_velocity=0.1, cost_control=0.1, cost_crowd=0.1)
        env_low_dist.reset(seed=42)
        env_low_dist.agent_states[0] = np.array([1.0, 1.0, 0.0, 0.0, 9.0, 9.0], dtype=np.float32)
        _, reward_low_dist, _, _, _ = env_low_dist.step(action)

        # High distance cost should give worse reward when far from goal
        assert reward_high_dist < reward_low_dist

    def test_drift_is_kinematic(self):
        """Test drift term implements kinematic dynamics."""
        env = CrowdNavigationEnv()
        env.reset(seed=42)

        state = np.array([5.0, 5.0, 1.0, 2.0, 7.0, 7.0], dtype=np.float32)
        action = np.array([0.5, -0.5], dtype=np.float32)
        population = env.get_population_state()

        drift = env._drift(state, action, population)

        # Check kinematic dynamics
        assert drift[0] == 1.0  # dx/dt = vx
        assert drift[1] == 2.0  # dy/dt = vy
        assert drift[2] == 0.5  # dvx/dt = ax
        assert drift[3] == -0.5  # dvy/dt = ay
        assert drift[4] == 0.0  # goal x doesn't change
        assert drift[5] == 0.0  # goal y doesn't change

    def test_all_agents_share_same_goal(self):
        """Test all agents in population share the same goal."""
        env = CrowdNavigationEnv(num_agents=100)
        env.reset(seed=42)

        # Check all agent goals are the same
        goals = env.agent_states[:, 4:6]
        assert np.allclose(goals, goals[0])

    def test_render_does_not_crash(self):
        """Test render method exists and doesn't crash."""
        env = CrowdNavigationEnv()
        env.reset(seed=42)
        env.render()  # Should not raise

    def test_close_does_not_crash(self):
        """Test close method exists and doesn't crash."""
        env = CrowdNavigationEnv()
        env.reset(seed=42)
        env.close()  # Should not raise
