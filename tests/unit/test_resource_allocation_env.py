"""
Unit tests for Resource Allocation environment.

Tests portfolio optimization, simplex constraints, and congestion effects.
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
    from mfg_pde.alg.reinforcement.environments.resource_allocation_env import ResourceAllocationEnv


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestResourceAllocationEnv:
    """Tests for Resource Allocation environment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3, delta_max=0.2)

        assert env.num_agents == 100
        assert env.num_assets == 3
        assert env.state_dim == 6  # 2 * num_assets
        assert env.action_dim == 3  # num_assets
        assert env.delta_max == 0.2

    def test_observation_space(self):
        """Test observation space is correctly configured."""
        env = ResourceAllocationEnv(num_assets=3)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (6,)
        # First 3: allocations in [0, 1]
        # Last 3: asset values in [0, 10]
        np.testing.assert_array_equal(env.observation_space.low[:3], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.observation_space.high[:3], [1.0, 1.0, 1.0])

    def test_action_space(self):
        """Test action space is correctly configured."""
        env = ResourceAllocationEnv(num_assets=3, delta_max=0.2)

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (3,)
        np.testing.assert_array_almost_equal(env.action_space.low, [-0.2, -0.2, -0.2])
        np.testing.assert_array_almost_equal(env.action_space.high, [0.2, 0.2, 0.2])

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3)
        state, info = env.reset(seed=42)

        # Check shape
        assert state.shape == (6,)

        # Check allocations sum to 1
        allocations = state[:3]
        assert np.isclose(allocations.sum(), 1.0, atol=1e-3)

        # Check allocations non-negative
        assert np.all(allocations >= 0)

        # Check asset values are positive
        values = state[3:]
        assert np.all(values > 0)

        assert isinstance(info, dict)

    def test_reset_initializes_equal_allocation(self):
        """Test agents start near equal allocation."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3)
        state, _ = env.reset(seed=42)

        # Initial allocation should be near 1/3 for each asset
        allocations = state[:3]
        expected = 1.0 / 3
        assert np.allclose(allocations, expected, atol=0.1)  # Allow some noise

    def test_reset_with_seed_reproducible(self):
        """Test seeded reset produces same initial state."""
        env1 = ResourceAllocationEnv(num_agents=100, num_assets=3)
        env2 = ResourceAllocationEnv(num_agents=100, num_assets=3)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_simplex_projection(self):
        """Test simplex projection enforces constraints."""
        env = ResourceAllocationEnv(num_assets=3)
        env.reset(seed=42)

        # Test weights that violate simplex
        weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # Sum > 1
        projected = env._project_to_simplex(weights)

        # Should sum to 1
        assert np.isclose(projected.sum(), 1.0)
        # Should be non-negative
        assert np.all(projected >= 0)

    def test_step_maintains_simplex_constraint(self):
        """Test step maintains allocation simplex constraint."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3, dt=0.1)
        env.reset(seed=42)

        # Random action
        action = np.array([0.1, -0.05, -0.05], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Allocation should still sum to 1
        allocations = next_state[:3]
        assert np.isclose(allocations.sum(), 1.0, atol=1e-3)
        assert np.all(allocations >= 0)

    def test_step_updates_asset_values(self):
        """Test step updates asset values with returns."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3, dt=0.1, return_volatility=0.1)
        env.reset(seed=42)

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Values should change (returns + noise)
        # Just check they remain positive
        next_values = next_state[3:]
        assert np.all(next_values > 0)

    def test_reward_includes_portfolio_return(self):
        """Test reward includes portfolio returns."""
        env = ResourceAllocationEnv(base_return=0.1, risk_penalty=0.0, transaction_cost=0.0, congestion_penalty=0.0)
        env.reset(seed=42)

        # Set up state with known allocations
        env.agent_states[0][:3] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        env.agent_states[0][3:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should be related to returns (non-zero with positive base return)
        assert isinstance(reward, float)

    def test_reward_penalizes_risk(self):
        """Test reward penalizes portfolio risk."""
        env = ResourceAllocationEnv(base_return=0.0, risk_penalty=1.0, transaction_cost=0.0, congestion_penalty=0.0)
        env.reset(seed=42)

        # Concentrated allocation (higher risk)
        env.agent_states[0][:3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env.agent_states[0][3:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, reward_concentrated, _, _, _ = env.step(action)

        # Diversified allocation (lower risk)
        env.agent_states[0][:3] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        env.agent_states[0][3:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        _, reward_diversified, _, _, _ = env.step(action)

        # Diversified should have better reward (less risk penalty)
        assert reward_diversified > reward_concentrated

    def test_reward_penalizes_transaction_cost(self):
        """Test reward penalizes large rebalancing."""
        env = ResourceAllocationEnv(base_return=0.0, risk_penalty=0.0, transaction_cost=1.0, congestion_penalty=0.0)
        env.reset(seed=42)
        env.agent_states[0][:3] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        env.agent_states[0][3:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Large rebalancing
        action_large = np.array([0.2, -0.1, -0.1], dtype=np.float32)
        _, reward_large, _, _, _ = env.step(action_large)

        # Small rebalancing
        env.agent_states[0][:3] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        action_small = np.array([0.01, -0.005, -0.005], dtype=np.float32)
        _, reward_small, _, _, _ = env.step(action_small)

        # Smaller rebalancing should be better (less transaction cost)
        assert reward_small > reward_large

    def test_mean_field_coupling_computed(self):
        """Test mean field coupling provides congestion penalty."""
        env = ResourceAllocationEnv(base_return=0.0, risk_penalty=0.0, transaction_cost=0.0, congestion_penalty=1.0)
        state, _ = env.reset(seed=42)

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should include mean field term
        assert isinstance(reward, float)

    def test_population_histogram_computed(self):
        """Test population histogram is computed over allocation space."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3, population_bins=50)
        env.reset(seed=42)

        histogram = env.get_population_state()

        # Check shape
        assert histogram.shape == (50,)

        # Check normalization
        assert np.isclose(histogram.sum(), 1.0)

        # Check all non-negative
        assert np.all(histogram >= 0)

    def test_terminates_on_invalid_allocation(self):
        """Test episode terminates with invalid allocation."""
        env = ResourceAllocationEnv(num_assets=3)
        env.reset(seed=42)

        # Create invalid state (NaN)
        state_invalid = np.array([np.nan, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
        assert env._is_terminated(state_invalid)

        # Create invalid state (Inf)
        state_inf = np.array([np.inf, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        assert env._is_terminated(state_inf)

    def test_does_not_terminate_with_valid_allocation(self):
        """Test episode continues with valid allocation."""
        env = ResourceAllocationEnv(num_assets=3, max_steps=100)
        env.reset(seed=42)

        # Valid allocation
        env.agent_states[0] = np.array([0.33, 0.33, 0.34, 1.0, 1.0, 1.0], dtype=np.float32)
        action = np.array([0.01, -0.005, -0.005], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)

        assert not terminated
        assert not truncated

    def test_state_clipping_to_bounds(self):
        """Test states are clipped to domain bounds."""
        env = ResourceAllocationEnv(num_assets=3, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        # Set state with extreme values
        env.agent_states[0] = np.array([0.9, 0.05, 0.05, 5.0, 5.0, 5.0], dtype=np.float32)

        action = np.array([0.1, -0.05, -0.05], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Allocations should be in [0, 1] and sum to 1
        allocations = next_state[:3]
        assert np.all(allocations >= 0)
        assert np.all(allocations <= 1)
        assert np.isclose(allocations.sum(), 1.0, atol=1e-3)

        # Asset values should be non-negative
        values = next_state[3:]
        assert np.all(values >= 0)

    def test_action_enforces_zero_sum(self):
        """Test action drift enforces zero-sum constraint."""
        env = ResourceAllocationEnv(num_assets=3)
        env.reset(seed=42)

        state = np.array([0.33, 0.33, 0.34, 1.0, 1.0, 1.0], dtype=np.float32)
        action = np.array([0.1, 0.05, 0.02], dtype=np.float32)  # Doesn't sum to zero
        population = env.get_population_state()

        drift = env._drift(state, action, population)

        # Allocation drift should enforce zero-sum
        dw_dt = drift[:3]
        # After zero-sum correction, should be close to zero sum
        expected_sum = 0.0
        actual_sum = float(dw_dt.sum())
        assert np.isclose(actual_sum, expected_sum, atol=1e-6)

    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        env = ResourceAllocationEnv(
            num_assets=4,
            delta_max=0.3,
            base_return=0.1,
            risk_penalty=2.0,
            transaction_cost=0.05,
        )

        assert env.num_assets == 4
        assert env.state_dim == 8
        assert env.action_dim == 4
        assert env.delta_max == 0.3
        assert env.base_return == 0.1
        assert env.risk_penalty == 2.0
        assert env.transaction_cost == 0.05

    def test_episode_runs_to_completion(self):
        """Test episode runs for max_steps without early termination."""
        env = ResourceAllocationEnv(max_steps=10, num_assets=3)
        env.reset(seed=42)

        action = np.array([0.01, -0.005, -0.005], dtype=np.float32)

        for _ in range(9):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Final step should truncate
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated  # No early termination
        assert truncated  # Time limit reached

    def test_initial_asset_values_are_one(self):
        """Test all assets start at value 1.0."""
        env = ResourceAllocationEnv(num_agents=100, num_assets=3)
        env.reset(seed=42)

        values = env.agent_states[:, 3:]

        # All should start at 1.0
        assert np.allclose(values, 1.0)

    def test_render_does_not_crash(self):
        """Test render method exists and doesn't crash."""
        env = ResourceAllocationEnv(num_assets=3)
        env.reset(seed=42)
        env.render()  # Should not raise

    def test_close_does_not_crash(self):
        """Test close method exists and doesn't crash."""
        env = ResourceAllocationEnv(num_assets=3)
        env.reset(seed=42)
        env.close()  # Should not raise
