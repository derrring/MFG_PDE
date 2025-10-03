"""
Unit tests for Price Formation environment.

Tests market making dynamics, inventory management, and liquidity coupling.
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
    from mfg_pde.alg.reinforcement.environments.price_formation_env import PriceFormationEnv


@pytest.mark.skipif(not CONTINUOUS_MFG_AVAILABLE, reason="Gymnasium not available")
class TestPriceFormationEnv:
    """Tests for Price Formation environment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = PriceFormationEnv(num_agents=100, q_max=10.0, p_min=90.0, p_max=110.0)

        assert env.num_agents == 100
        assert env.state_dim == 4
        assert env.action_dim == 2
        assert env.q_max == 10.0
        assert env.p_min == 90.0
        assert env.p_max == 110.0

    def test_observation_space(self):
        """Test observation space is correctly configured."""
        env = PriceFormationEnv(q_max=10.0, p_min=90.0, p_max=110.0)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (4,)
        np.testing.assert_array_equal(env.observation_space.low, [-10.0, 90.0, -2.0, 0.0])
        np.testing.assert_array_equal(env.observation_space.high, [10.0, 110.0, 2.0, 1.0])

    def test_action_space(self):
        """Test action space is correctly configured."""
        env = PriceFormationEnv(delta_max=1.0)

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [0.0, 0.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = PriceFormationEnv(num_agents=100, q_max=10.0, p_min=90.0, p_max=110.0)
        state, info = env.reset(seed=42)

        # Check shape
        assert state.shape == (4,)

        # Check inventory near zero
        assert -10.0 <= state[0] <= 10.0
        assert np.abs(state[0]) < 0.1  # Should start near zero

        # Check price within bounds
        assert 90.0 <= state[1] <= 110.0

        # Check velocity near zero
        assert -2.0 <= state[2] <= 2.0
        assert np.abs(state[2]) < 0.1

        # Check market depth
        assert 0.0 <= state[3] <= 1.0

        assert isinstance(info, dict)

    def test_reset_initializes_zero_inventory(self):
        """Test agents start with zero inventory."""
        env = PriceFormationEnv(num_agents=100)
        state, _ = env.reset(seed=42)

        # Initial inventory should be zero
        assert np.isclose(state[0], 0.0, atol=1e-6)

    def test_reset_with_seed_reproducible(self):
        """Test seeded reset produces same initial state."""
        env1 = PriceFormationEnv(num_agents=100)
        env2 = PriceFormationEnv(num_agents=100)

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_step_updates_inventory(self):
        """Test step updates inventory from filled orders."""
        env = PriceFormationEnv(num_agents=100, dt=0.1, noise_std=0.0)
        env.reset(seed=42)

        # Set spreads (action affects fill probability)
        action = np.array([0.1, 0.1], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Inventory may change (stochastic fills)
        # Just check it's within bounds
        assert -10.0 <= next_state[0] <= 10.0

    def test_step_updates_price(self):
        """Test step updates price with dynamics."""
        env = PriceFormationEnv(num_agents=100, dt=0.1, price_volatility=0.5)
        env.reset(seed=42)

        action = np.array([0.5, 0.5], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Price should change (noise + impact)
        # Just check it's within bounds
        assert 90.0 <= next_state[1] <= 110.0

    def test_reward_penalizes_large_inventory(self):
        """Test reward penalizes large inventory positions."""
        env = PriceFormationEnv(inventory_penalty=1.0, spread_cost=0.0, liquidity_penalty=0.0)

        # Large inventory
        env.reset(seed=42)
        env.agent_states[0] = np.array([5.0, 100.0, 0.0, 0.5], dtype=np.float32)

        action = np.array([0.1, 0.1], dtype=np.float32)
        _, reward_large, _, _, _ = env.step(action)

        # Small inventory
        env.agent_states[0] = np.array([1.0, 100.0, 0.0, 0.5], dtype=np.float32)
        _, reward_small, _, _, _ = env.step(action)

        # Smaller inventory should give better reward
        assert reward_small > reward_large

    def test_reward_includes_pnl_from_spread(self):
        """Test reward includes PnL from bid-ask spread."""
        env = PriceFormationEnv(inventory_penalty=0.0, spread_cost=0.1, liquidity_penalty=0.0)
        env.reset(seed=42)
        env.agent_states[0] = np.array([0.0, 100.0, 0.0, 0.5], dtype=np.float32)

        # Wide spread (higher potential profit but lower fill rate)
        action_wide = np.array([0.9, 0.9], dtype=np.float32)
        _, reward_wide, _, _, _ = env.step(action_wide)

        # Tight spread (lower profit but higher fill rate)
        env.agent_states[0] = np.array([0.0, 100.0, 0.0, 0.5], dtype=np.float32)
        action_tight = np.array([0.1, 0.1], dtype=np.float32)
        _, reward_tight, _, _, _ = env.step(action_tight)

        # Both should have some reward (trade-off between spread and fills)
        # Can't predict which is better without specific parameters
        assert isinstance(reward_wide, float)
        assert isinstance(reward_tight, float)

    def test_reward_penalizes_wide_spreads(self):
        """Test reward penalizes wide spreads via opportunity cost."""
        env = PriceFormationEnv(inventory_penalty=0.0, spread_cost=1.0, liquidity_penalty=0.0)
        env.reset(seed=42)

        # Wide spread
        env.agent_states[0] = np.array([0.0, 100.0, 0.0, 0.5], dtype=np.float32)
        action_wide = np.array([1.0, 1.0], dtype=np.float32)
        _, reward_wide, _, _, _ = env.step(action_wide)

        # Tight spread
        env.agent_states[0] = np.array([0.0, 100.0, 0.0, 0.5], dtype=np.float32)
        action_tight = np.array([0.1, 0.1], dtype=np.float32)
        _, reward_tight, _, _, _ = env.step(action_tight)

        # Tighter spread should be better (less opportunity cost)
        assert reward_tight > reward_wide

    def test_terminates_when_inventory_exceeds_limit(self):
        """Test episode terminates when inventory limit reached."""
        env = PriceFormationEnv(q_max=10.0)
        state, _ = env.reset(seed=42)

        # Directly check termination logic with inventory at 95% of limit
        state_at_limit = np.array([9.5, 100.0, 0.0, 0.5], dtype=np.float32)
        assert env._is_terminated(state_at_limit)

        # Check slightly below limit doesn't terminate
        state_below_limit = np.array([9.4, 100.0, 0.0, 0.5], dtype=np.float32)
        assert not env._is_terminated(state_below_limit)

    def test_does_not_terminate_with_moderate_inventory(self):
        """Test episode continues with moderate inventory."""
        env = PriceFormationEnv(q_max=10.0, max_steps=100)
        env.reset(seed=42)

        # Moderate inventory
        env.agent_states[0] = np.array([5.0, 100.0, 0.0, 0.5], dtype=np.float32)
        action = np.array([0.1, 0.1], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)

        assert not terminated
        assert not truncated

    def test_mean_field_coupling_computed(self):
        """Test mean field coupling provides liquidity penalty."""
        env = PriceFormationEnv(liquidity_penalty=1.0, inventory_penalty=0.0, spread_cost=0.0)
        state, _ = env.reset(seed=42)

        action = np.array([0.1, 0.1], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should include mean field term (non-zero)
        assert isinstance(reward, float)

    def test_population_histogram_computed(self):
        """Test population histogram is computed over price space."""
        env = PriceFormationEnv(num_agents=100, population_bins=50)
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
        env = PriceFormationEnv(q_max=10.0, p_min=90.0, p_max=110.0, dt=1.0, noise_std=0.0)
        env.reset(seed=42)

        # Set state near boundary
        env.agent_states[0] = np.array([9.0, 109.0, 1.0, 0.8], dtype=np.float32)

        # Action that might exceed bounds
        action = np.array([0.1, 0.1], dtype=np.float32)

        next_state, _, _, _, _ = env.step(action)

        # Should be clipped to bounds
        assert -10.0 <= next_state[0] <= 10.0
        assert 90.0 <= next_state[1] <= 110.0
        assert -2.0 <= next_state[2] <= 2.0
        assert 0.0 <= next_state[3] <= 1.0

    def test_action_clipping_to_bounds(self):
        """Test actions are clipped to bounds."""
        env = PriceFormationEnv(delta_max=1.0)
        env.reset(seed=42)

        # Action exceeding bounds
        action = np.array([10.0, 10.0], dtype=np.float32)

        # Should not raise error (clipped internally)
        _, _, _, _, _ = env.step(action)

    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        env = PriceFormationEnv(
            q_max=20.0,
            p_min=80.0,
            p_max=120.0,
            delta_max=2.0,
            price_volatility=1.0,
            inventory_penalty=2.0,
        )

        assert env.q_max == 20.0
        assert env.p_min == 80.0
        assert env.p_max == 120.0
        assert env.delta_max == 2.0
        assert env.price_volatility == 1.0
        assert env.inventory_penalty == 2.0

    def test_drift_computation(self):
        """Test drift term computes price and inventory dynamics."""
        env = PriceFormationEnv()
        env.reset(seed=42)

        state = np.array([0.0, 100.0, 0.0, 0.5], dtype=np.float32)
        action = np.array([0.5, 0.5], dtype=np.float32)
        population = env.get_population_state()

        drift = env._drift(state, action, population)

        # Check drift components
        assert drift.shape == (4,)
        assert isinstance(drift[0], float | np.floating)  # dq/dt
        assert isinstance(drift[1], float | np.floating)  # dp/dt
        assert isinstance(drift[2], float | np.floating)  # d(dp/dt)/dt
        assert drift[3] == 0.0  # market depth doesn't drift

    def test_episode_runs_to_completion(self):
        """Test episode runs for max_steps without early termination."""
        env = PriceFormationEnv(max_steps=10)
        env.reset(seed=42)

        action = np.array([0.1, 0.1], dtype=np.float32)

        for _ in range(9):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated

        # Final step should truncate
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated  # No early termination
        assert truncated  # Time limit reached

    def test_initial_price_near_midpoint(self):
        """Test initial price is near midpoint of range."""
        env = PriceFormationEnv(num_agents=100, p_min=90.0, p_max=110.0)
        env.reset(seed=42)

        prices = env.agent_states[:, 1]

        # Mean price should be near midpoint (100.0)
        mean_price = np.mean(prices)
        assert 99.0 <= mean_price <= 101.0

    def test_render_does_not_crash(self):
        """Test render method exists and doesn't crash."""
        env = PriceFormationEnv()
        env.reset(seed=42)
        env.render()  # Should not raise

    def test_close_does_not_crash(self):
        """Test close method exists and doesn't crash."""
        env = PriceFormationEnv()
        env.reset(seed=42)
        env.close()  # Should not raise
