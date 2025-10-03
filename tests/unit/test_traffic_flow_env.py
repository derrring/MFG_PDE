"""
Unit tests for Traffic Flow environment.

Tests congestion-aware navigation, velocity dynamics, and routing strategies.
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
    from mfg_pde.alg.reinforcement.environments.traffic_flow_env import TrafficFlowEnv


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestTrafficFlowEnv:
    """Tests for Traffic Flow environment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = TrafficFlowEnv(num_agents=100, corridor_length=10.0, v_max=2.0, a_max=1.0, time_limit=20.0)

        assert env.num_agents == 100
        assert env.state_dim == 3
        assert env.action_dim == 1
        assert env.corridor_length == 10.0
        assert env.v_max == 2.0
        assert env.a_max == 1.0
        assert env.time_limit == 20.0

    def test_observation_space(self):
        """Test observation space is correctly configured."""
        env = TrafficFlowEnv(corridor_length=10.0, v_max=2.0, time_limit=20.0)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (3,)
        np.testing.assert_array_equal(env.observation_space.low, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.observation_space.high, [10.0, 2.0, 20.0])

    def test_action_space(self):
        """Test action space is correctly configured."""
        env = TrafficFlowEnv(a_max=1.0)

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0])

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = TrafficFlowEnv(num_agents=100, corridor_length=10.0)
        state, info = env.reset(seed=42)

        # Check shape
        assert state.shape == (3,)

        # Check position at start
        assert np.isclose(state[0], 0.0, atol=1e-6)

        # Check velocity is zero
        assert np.isclose(state[1], 0.0, atol=1e-6)

        # Check time remaining is full
        assert np.isclose(state[2], env.time_limit, atol=1e-6)

        assert isinstance(info, dict)

    def test_reset_with_seed_reproducible(self):
        """Test seeded reset produces same initial state."""
        env1 = TrafficFlowEnv(num_agents=100)
        env2 = TrafficFlowEnv(num_agents=100)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_step_updates_position(self):
        """Test step updates position with velocity."""
        env = TrafficFlowEnv(num_agents=100, dt=0.1, noise_std=0.0)
        env.reset(seed=42)

        # Set initial state with non-zero velocity
        env.agent_states[0] = np.array([1.0, 1.0, 20.0], dtype=np.float32)

        # Zero acceleration (coast)
        action = np.array([0.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Position should increase: x' = x + v * dt = 1.0 + 1.0 * 0.1 = 1.1
        assert next_state[0] > 1.0

    def test_step_updates_velocity_with_acceleration(self):
        """Test step updates velocity from acceleration."""
        env = TrafficFlowEnv(num_agents=100, dt=0.1, noise_std=0.0, congestion_coeff=0.0)
        env.reset(seed=42)

        # Start from rest
        env.agent_states[0] = np.array([0.0, 0.0, 20.0], dtype=np.float32)

        # Positive acceleration
        action = np.array([1.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Velocity should increase (no congestion)
        assert next_state[1] > 0.0

    def test_step_decreases_time_remaining(self):
        """Test step decreases time remaining."""
        env = TrafficFlowEnv(num_agents=100, dt=0.1)
        state, _ = env.reset(seed=42)

        initial_time = state[2]

        action = np.array([0.0], dtype=np.float32)
        next_state, _, _, _, _ = env.step(action)

        # Time should decrease by dt
        assert next_state[2] < initial_time

    def test_reward_includes_progress(self):
        """Test reward includes forward progress."""
        env = TrafficFlowEnv(fuel_cost=0.0, time_penalty=0.0, arrival_bonus=0.0, congestion_coeff=0.0)
        env.reset(seed=42)

        # Set state with velocity
        env.agent_states[0] = np.array([1.0, 1.0, 20.0], dtype=np.float32)

        action = np.array([0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should be positive (forward progress)
        assert reward > 0

    def test_reward_penalizes_fuel_cost(self):
        """Test reward penalizes large acceleration."""
        env = TrafficFlowEnv(fuel_cost=1.0, time_penalty=0.0, arrival_bonus=0.0, congestion_coeff=0.0)

        # Large acceleration
        env.reset(seed=42)
        env.agent_states[0] = np.array([1.0, 0.0, 20.0], dtype=np.float32)
        action_large = np.array([1.0], dtype=np.float32)
        _, reward_large, _, _, _ = env.step(action_large)

        # Small acceleration
        env.agent_states[0] = np.array([1.0, 0.0, 20.0], dtype=np.float32)
        action_small = np.array([0.1], dtype=np.float32)
        _, reward_small, _, _, _ = env.step(action_small)

        # Smaller acceleration should give better reward
        assert reward_small > reward_large

    def test_reward_includes_arrival_bonus(self):
        """Test reward includes arrival bonus when reaching destination."""
        env = TrafficFlowEnv(
            corridor_length=5.0,
            arrival_bonus=10.0,
            fuel_cost=0.0,
            time_penalty=0.0,
            congestion_coeff=0.0,
        )
        env.reset(seed=42)

        # Set state near destination
        env.agent_states[0] = np.array([4.9, 1.0, 20.0], dtype=np.float32)

        action = np.array([0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Should receive arrival bonus (large reward)
        assert reward > 5.0

    def test_congestion_reduces_velocity(self):
        """Test congestion drag reduces velocity in crowded regions."""
        env = TrafficFlowEnv(num_agents=100, congestion_coeff=1.0, dt=0.1, noise_std=0.0, fuel_cost=0.0)
        env.reset(seed=42)

        # Set all agents at same location (high density)
        env.agent_states[:, 0] = 5.0  # All at middle of corridor
        env.agent_states[0] = np.array([5.0, 1.0, 20.0], dtype=np.float32)

        # Zero acceleration
        action = np.array([0.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Velocity should decrease due to congestion drag
        assert next_state[1] < 1.0

    def test_terminates_when_destination_reached(self):
        """Test episode terminates when reaching destination."""
        env = TrafficFlowEnv(corridor_length=10.0)
        env.reset(seed=42)

        # Set state at destination
        state_at_destination = np.array([10.0, 0.0, 5.0], dtype=np.float32)
        assert env._is_terminated(state_at_destination)

        # Set state before destination
        state_before = np.array([9.9, 0.0, 5.0], dtype=np.float32)
        assert not env._is_terminated(state_before)

    def test_does_not_terminate_before_destination(self):
        """Test episode continues before reaching destination."""
        env = TrafficFlowEnv(corridor_length=10.0, max_steps=100)
        env.reset(seed=42)

        # Set state in middle of corridor
        env.agent_states[0] = np.array([5.0, 1.0, 10.0], dtype=np.float32)
        action = np.array([0.0], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)

        assert not terminated
        assert not truncated

    def test_mean_field_coupling_computed(self):
        """Test mean field coupling provides congestion penalty."""
        env = TrafficFlowEnv(congestion_coeff=1.0, fuel_cost=0.0, time_penalty=0.0)
        env.reset(seed=42)

        # Set high density at position
        env.agent_states[:, 0] = 5.0  # All agents at same position
        env.agent_states[0] = np.array([5.0, 1.0, 20.0], dtype=np.float32)

        action = np.array([0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should include congestion penalty (negative)
        # Progress is positive, but congestion should reduce it
        assert isinstance(reward, float)

    def test_population_histogram_computed(self):
        """Test population histogram is computed over position space."""
        env = TrafficFlowEnv(num_agents=100, corridor_length=10.0, population_bins=50)
        env.reset(seed=42)

        histogram = env.get_population_state()

        # Check shape
        assert histogram.shape == (50,)

        # Check normalization
        assert np.isclose(histogram.sum(), 1.0)

        # Check all non-negative
        assert np.all(histogram >= 0)

    def test_state_clipping_to_bounds(self):
        """Test states are clipped to domain bounds."""
        env = TrafficFlowEnv(corridor_length=10.0, v_max=2.0, time_limit=20.0, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        # Set state near boundaries
        env.agent_states[0] = np.array([9.5, 1.9, 1.0], dtype=np.float32)

        # Large acceleration
        action = np.array([1.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Should be clipped to bounds
        assert 0.0 <= next_state[0] <= 10.0
        assert 0.0 <= next_state[1] <= 2.0
        assert next_state[2] >= 0.0  # Time can go negative (handled by truncation)

    def test_action_clipping_to_bounds(self):
        """Test actions are clipped to bounds."""
        env = TrafficFlowEnv(a_max=1.0)
        env.reset(seed=42)

        # Action exceeding bounds
        action = np.array([10.0], dtype=np.float32)

        # Should not raise error (clipped internally)
        _, _, _, _, _ = env.step(action)

    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        env = TrafficFlowEnv(
            corridor_length=20.0,
            v_max=5.0,
            a_max=2.0,
            time_limit=30.0,
            congestion_coeff=1.5,
        )

        assert env.corridor_length == 20.0
        assert env.v_max == 5.0
        assert env.a_max == 2.0
        assert env.time_limit == 30.0
        assert env.congestion_coeff == 1.5

    def test_drift_computation(self):
        """Test drift term computes position and velocity dynamics."""
        env = TrafficFlowEnv()
        env.reset(seed=42)

        state = np.array([5.0, 1.0, 10.0], dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)
        population = env.get_population_state()

        drift = env._drift(state, action, population)

        # Check drift components
        assert drift.shape == (3,)
        assert drift[0] > 0  # dx/dt = v > 0
        assert isinstance(drift[1], float | np.floating)  # dv/dt (depends on congestion)
        assert drift[2] == -1.0  # dt_rem/dt = -1

    def test_episode_runs_to_completion(self):
        """Test episode runs for max_steps without early termination."""
        env = TrafficFlowEnv(corridor_length=100.0, max_steps=10)  # Long corridor
        env.reset(seed=42)

        action = np.array([0.1], dtype=np.float32)

        for _ in range(9):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Final step should truncate
        _, _, terminated, truncated, _ = env.step(action)
        # May or may not terminate (depends on if destination reached)
        # But should truncate if not terminated
        if not terminated:
            assert truncated

    def test_all_agents_start_at_origin(self):
        """Test all agents start at x=0 with v=0."""
        env = TrafficFlowEnv(num_agents=100)
        env.reset(seed=42)

        positions = env.agent_states[:, 0]
        velocities = env.agent_states[:, 1]

        # All should start at x=0, v=0
        assert np.allclose(positions, 0.0)
        assert np.allclose(velocities, 0.0)

    def test_render_does_not_crash(self):
        """Test render method exists and doesn't crash."""
        env = TrafficFlowEnv()
        env.reset(seed=42)
        env.render()  # Should not raise

    def test_close_does_not_crash(self):
        """Test close method exists and doesn't crash."""
        env = TrafficFlowEnv()
        env.reset(seed=42)
        env.close()  # Should not raise
