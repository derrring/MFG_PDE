"""
Basic unit tests for multi-population algorithms (DDPG, TD3, SAC).

Tests basic functionality: initialization, imports, and API consistency.
"""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms import (
        MultiPopulationDDPG,
        MultiPopulationSAC,
        MultiPopulationTD3,
    )
from tests.unit.test_multi_population_env import SimpleMultiPopEnv


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiPopulationAlgorithmsBasic:
    """Basic tests for all multi-population algorithms."""

    def setup_method(self):
        """Create a simple 2-population environment for testing."""
        self.env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
            ],
            population_sizes=50,
            max_steps=10,
        )
        self.env.reset(seed=42)

    def test_multi_population_ddpg_initialization(self):
        """Test MultiPopulationDDPG can be initialized."""
        algo = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        assert algo.num_populations == 2
        assert len(algo.actors) == 2
        assert len(algo.critics) == 2
        assert len(algo.replay_buffers) == 2

    def test_multi_population_td3_initialization(self):
        """Test MultiPopulationTD3 can be initialized."""
        algo = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        assert algo.num_populations == 2
        assert len(algo.actors) == 2
        assert len(algo.critics1) == 2  # Twin critics
        assert len(algo.critics2) == 2

    def test_multi_population_sac_initialization(self):
        """Test MultiPopulationSAC can be initialized."""
        algo = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        assert algo.num_populations == 2
        assert len(algo.actors) == 2
        assert len(algo.critics1) == 2
        assert len(algo.alphas) == 2  # Per-population temperature

    def test_heterogeneous_dimensions(self):
        """Test all algorithms work with heterogeneous dimensions."""
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

        for algo_class in [MultiPopulationDDPG, MultiPopulationTD3, MultiPopulationSAC]:
            algo = algo_class(
                env=env,
                num_populations=3,
                state_dims=[2, 3, 1],
                action_dims=[2, 3, 1],
                population_dims=[50, 100, 30],
                action_bounds=[(-1, 1), (-2, 2), (0, 1)],
            )
            assert algo.num_populations == 3
            assert algo.state_dims == [2, 3, 1]
            assert algo.action_dims == [2, 3, 1]

    def test_ddpg_action_selection(self):
        """Test DDPG can select actions for all populations."""
        algo = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        actions = algo.select_actions(states, pop_states, training=False)

        assert len(actions) == 2
        assert 0 in actions
        assert 1 in actions
        assert actions[0].shape == (2,)
        assert actions[1].shape == (2,)

    def test_td3_action_selection(self):
        """Test TD3 can select actions for all populations."""
        algo = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        actions = algo.select_actions(states, pop_states, training=False)

        assert len(actions) == 2
        assert actions[0].shape == (2,)
        assert actions[1].shape == (2,)

    def test_sac_action_selection(self):
        """Test SAC can select actions for all populations."""
        algo = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Test stochastic sampling (training)
        actions_train = algo.select_actions(states, pop_states, training=True)
        assert len(actions_train) == 2

        # Test deterministic (evaluation)
        actions_eval = algo.select_actions(states, pop_states, training=False)
        assert len(actions_eval) == 2

    def test_action_bounds_respected(self):
        """Test that selected actions respect bounds."""
        algo = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-0.5, 0.5)],  # Different bounds
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        actions = algo.select_actions(states, pop_states, training=False)

        # Check bounds
        assert (actions[0] >= -1).all()
        assert (actions[0] <= 1).all()
        assert (actions[1] >= -0.5).all()
        assert (actions[1] <= 0.5).all()

    def test_config_override(self):
        """Test that config can be overridden."""
        custom_config = {
            "actor_lr": 1e-3,
            "critic_lr": 5e-3,
            "batch_size": 128,
        }

        algo = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=custom_config,
        )

        assert algo.config["actor_lr"] == 1e-3
        assert algo.config["critic_lr"] == 5e-3
        assert algo.config["batch_size"] == 128

    def test_algorithms_fail_single_population(self):
        """Test all algorithms fail with N < 2."""
        for algo_class in [MultiPopulationDDPG, MultiPopulationTD3, MultiPopulationSAC]:
            with pytest.raises(ValueError, match="Multi-population requires N ≥ 2"):
                algo_class(
                    env=self.env,
                    num_populations=1,
                    state_dims=2,
                    action_dims=[2],
                    population_dims=50,
                    action_bounds=[(-1, 1)],
                )
