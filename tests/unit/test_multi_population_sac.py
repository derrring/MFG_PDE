"""
Unit tests for Multi-Population SAC algorithm.

Tests stochastic policies, entropy regularization, automatic temperature tuning,
and reparameterization trick.
"""

from __future__ import annotations

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

from mfg_pde.alg.reinforcement.algorithms import MultiPopulationSAC
from tests.unit.test_multi_population_env import SimpleMultiPopEnv


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiPopulationSAC:
    """Comprehensive tests for MultiPopulationSAC."""

    def setup_method(self):
        """Create test environment and algorithm."""
        self.env = SimpleMultiPopEnv(
            num_populations=2,
            state_dims=2,
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
            ],
            population_sizes=50,
            max_steps=20,
        )
        self.env.reset(seed=42)

        # Use smaller batch size for faster testing
        test_config = {"batch_size": 32}

        self.algo = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=test_config,
        )

    def test_stochastic_policy_sampling(self):
        """Test policy produces different actions each time (stochastic)."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Sample multiple times in training mode
        actions_1 = self.algo.select_actions(states, pop_states, training=True)
        actions_2 = self.algo.select_actions(states, pop_states, training=True)
        actions_3 = self.algo.select_actions(states, pop_states, training=True)

        # Actions should differ (stochastic sampling)
        assert not np.allclose(actions_1[0], actions_2[0], atol=1e-6)
        assert not np.allclose(actions_2[0], actions_3[0], atol=1e-6)

    def test_deterministic_mean_in_eval(self):
        """Test policy uses mean action in eval mode (deterministic)."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Multiple eval samples should be identical
        actions_1 = self.algo.select_actions(states, pop_states, training=False)
        actions_2 = self.algo.select_actions(states, pop_states, training=False)

        assert np.allclose(actions_1[0], actions_2[0])
        assert np.allclose(actions_1[1], actions_2[1])

    def test_per_population_temperature_initialized(self):
        """Test each population has independent temperature parameter."""
        assert len(self.algo.alphas) == 2
        assert len(self.algo.log_alphas) == 2
        assert len(self.algo.alpha_optimizers) == 2

        # Alphas should be independent tensors
        assert self.algo.alphas[0] is not self.algo.alphas[1]

    def test_automatic_temperature_tuning(self):
        """Test temperature is tuned automatically during training."""
        initial_alphas = [self.algo.alphas[i] for i in range(2)]

        # Train and check alphas change
        self.algo.train(num_episodes=5)

        final_alphas = [self.algo.alphas[i] for i in range(2)]

        # At least one alpha should have changed
        assert not np.isclose(initial_alphas[0], final_alphas[0]) or not np.isclose(initial_alphas[1], final_alphas[1])

    def test_temperature_tuning_can_be_disabled(self):
        """Test temperature tuning can be disabled via config."""
        config = {"auto_tune_temperature": False, "initial_temperature": 0.5}

        algo_fixed_temp = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=config,
        )

        # Should have no alpha optimizers (list of None for each population)
        assert all(opt is None for opt in algo_fixed_temp.alpha_optimizers)

        # Temperature should remain fixed
        initial_alpha = algo_fixed_temp.alphas[0]
        algo_fixed_temp.train(num_episodes=3)
        final_alpha = algo_fixed_temp.alphas[0]

        assert np.isclose(initial_alpha, final_alpha)

    def test_entropy_in_training_stats(self):
        """Test entropy statistics are recorded during training."""
        stats = self.algo.train(num_episodes=3)

        assert "alphas" in stats
        assert len(stats["alphas"]) == 2

        for pop_id in range(2):
            assert len(stats["alphas"][pop_id]) > 0

    def test_twin_critics_for_sac(self):
        """Test SAC uses twin critics like TD3."""
        assert len(self.algo.critics1) == 2
        assert len(self.algo.critics2) == 2

        # Critics should be different
        assert self.algo.critics1[0] is not self.algo.critics2[0]

    def test_log_prob_computation_with_tanh(self):
        """Test log probability includes tanh correction term."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        state_tensor = torch.FloatTensor(states[0]).unsqueeze(0)
        pop_state_list = [pop_states[i] for i in range(2)]
        pop_state_tensor = torch.FloatTensor(np.concatenate(pop_state_list)).unsqueeze(0)

        # Sample action and log prob
        with torch.no_grad():
            action, log_prob, _ = self.algo.actors[0].sample(state_tensor, pop_state_tensor)

        # Log prob should be negative (valid probability)
        assert log_prob.item() <= 0

        # Action should be in bounds due to tanh
        assert torch.all(action >= -1 - 1e-5)
        assert torch.all(action <= 1 + 1e-5)

    def test_reparameterization_trick_gradients(self):
        """Test reparameterization allows gradient flow."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        state_tensor = torch.FloatTensor(states[0]).unsqueeze(0).requires_grad_(True)
        pop_state_list = [pop_states[i] for i in range(2)]
        pop_state_tensor = torch.FloatTensor(np.concatenate(pop_state_list)).unsqueeze(0).requires_grad_(True)

        # Sample action (reparameterization)
        action, _log_prob, _ = self.algo.actors[0].sample(state_tensor, pop_state_tensor)

        # Compute dummy loss and backprop
        loss = action.sum()
        loss.backward()

        # Gradients should flow to state
        assert state_tensor.grad is not None

    def test_target_entropy_per_population(self):
        """Test target entropy is computed per population."""
        assert len(self.algo.target_entropies) == 2

        # Target entropy should be -action_dim for each population
        assert np.isclose(self.algo.target_entropies[0], -2.0)
        assert np.isclose(self.algo.target_entropies[1], -2.0)

    def test_custom_target_entropy(self):
        """Test custom target entropy can be specified."""
        # Current implementation applies same target_entropy to all populations
        config = {"target_entropy": -1.5}

        algo_custom_entropy = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=config,
        )

        # Both populations should use the same custom target entropy
        assert np.isclose(algo_custom_entropy.target_entropies[0], -1.5)
        assert np.isclose(algo_custom_entropy.target_entropies[1], -1.5)

    def test_action_bounds_with_tanh_squashing(self):
        """Test actions are properly bounded with tanh squashing."""
        algo_bounded = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-0.5, 0.5), (-2, 2)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Sample many times
        for _ in range(50):
            actions = algo_bounded.select_actions(states, pop_states, training=True)

            # Check bounds
            assert np.all(actions[0] >= -0.5 - 1e-5)
            assert np.all(actions[0] <= 0.5 + 1e-5)
            assert np.all(actions[1] >= -2 - 1e-5)
            assert np.all(actions[1] <= 2 + 1e-5)

    def test_training_produces_all_losses(self):
        """Test training records critic and actor losses."""
        stats = self.algo.train(num_episodes=3)

        assert "critic1_losses" in stats
        assert "critic2_losses" in stats
        assert "actor_losses" in stats

        for pop_id in range(2):
            assert len(stats["critic1_losses"][pop_id]) > 0
            assert len(stats["critic2_losses"][pop_id]) > 0
            assert len(stats["actor_losses"][pop_id]) > 0

    def test_soft_update_target_critics(self):
        """Test target critics are soft-updated."""
        initial_params = [
            [p.clone() for p in self.algo.critic1_targets[0].parameters()],
            [p.clone() for p in self.algo.critic2_targets[0].parameters()],
        ]

        # Train
        self.algo.train(num_episodes=3)

        # Check targets updated
        for p_new, p_old in zip(self.algo.critic1_targets[0].parameters(), initial_params[0], strict=False):
            assert not torch.allclose(p_new, p_old, atol=1e-6)

        for p_new, p_old in zip(self.algo.critic2_targets[0].parameters(), initial_params[1], strict=False):
            assert not torch.allclose(p_new, p_old, atol=1e-6)

    def test_exploration_better_than_deterministic(self):
        """Test stochastic policy explores more than deterministic."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Collect multiple action samples
        train_actions = [self.algo.select_actions(states, pop_states, training=True) for _ in range(10)]

        # Compute variance
        train_variance = np.var([a[0] for a in train_actions], axis=0)

        # Should have non-zero variance (exploration)
        assert np.all(train_variance > 1e-6)

    def test_three_population_sac(self):
        """Test SAC works with 3 populations."""
        env_3pop = SimpleMultiPopEnv(
            num_populations=3,
            state_dims=[2, 3, 1],
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 3, "bounds": (-2, 2)},
                {"type": "continuous", "dim": 1, "bounds": (0, 1)},
            ],
            population_sizes=[50, 100, 30],
            max_steps=10,
        )

        algo_3pop = MultiPopulationSAC(
            env=env_3pop,
            num_populations=3,
            state_dims=[2, 3, 1],
            action_dims=[2, 3, 1],
            population_dims=[50, 100, 30],
            action_bounds=[(-1, 1), (-2, 2), (0, 1)],
        )

        stats = algo_3pop.train(num_episodes=2)

        assert len(stats["episode_rewards"]) == 3
        assert len(stats["alphas"]) == 3

    def test_replay_buffer_operations(self):
        """Test replay buffer works correctly with SAC."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()
        actions = self.algo.select_actions(states, pop_states, training=True)
        next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
        next_pop_states = self.env.get_population_states()

        # Flatten population states
        pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])
        next_pop_states_flat = np.concatenate([next_pop_states[i] for i in range(2)])

        # Fill buffer
        for _ in range(self.algo.config["batch_size"] + 10):
            self.algo.replay_buffers[0].push(
                states[0],
                actions[0],
                rewards[0],
                next_states[0],
                pop_states_flat,
                next_pop_states_flat,
                float(terminated[0]),
            )

        # Sample
        batch = self.algo.replay_buffers[0].sample(32)
        assert len(batch) == 7
        assert batch["states"].shape[0] == 32

    def test_initial_temperature_config(self):
        """Test initial temperature can be configured."""
        config = {"initial_temperature": 0.8}

        algo_high_temp = MultiPopulationSAC(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=config,
        )

        # Should start with specified temperature
        assert np.isclose(algo_high_temp.alphas[0], 0.8, atol=1e-3)

    def test_alpha_stays_positive(self):
        """Test temperature parameter remains positive during training."""
        self.algo.train(num_episodes=5)

        # All alphas should be positive
        for pop_id in range(2):
            assert self.algo.alphas[pop_id] > 0

    def test_mean_action_differs_from_sampled(self):
        """Test mean action differs from sampled action in training."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        state_tensor = torch.FloatTensor(states[0]).unsqueeze(0)
        pop_state_list = [pop_states[i] for i in range(2)]
        pop_state_tensor = torch.FloatTensor(np.concatenate(pop_state_list)).unsqueeze(0)

        with torch.no_grad():
            sampled_action, _, mean_action = self.algo.actors[0].sample(state_tensor, pop_state_tensor)

        # Sampled and mean should generally differ
        assert not torch.allclose(sampled_action, mean_action, atol=1e-3)
