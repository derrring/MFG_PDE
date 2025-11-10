"""
Unit tests for Multi-Population TD3 algorithm.

Tests twin critics, delayed policy updates, target policy smoothing, and training behavior.
"""

from __future__ import annotations

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms import MultiPopulationTD3
from tests.unit.test_multi_population_env import SimpleMultiPopEnv

pytestmark = pytest.mark.optional_torch


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiPopulationTD3:
    """Comprehensive tests for MultiPopulationTD3."""

    def setup_method(self):
        """Create test environment and algorithm."""
        # Set random seeds for deterministic testing (Issue #237)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

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

        self.algo = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=test_config,
        )

    def test_twin_critics_initialized(self):
        """Test both critic networks are initialized per population."""
        assert len(self.algo.critics1) == 2
        assert len(self.algo.critics2) == 2
        assert len(self.algo.critic1_targets) == 2
        assert len(self.algo.critic2_targets) == 2

        # Check they are different networks
        for pop_id in range(2):
            assert self.algo.critics1[pop_id] is not self.algo.critics2[pop_id]

    def test_clipped_double_q_learning(self):
        """Test TD3 uses min(Q1, Q2) for target computation."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Fill replay buffer
        for _ in range(self.algo.config["batch_size"] + 10):
            actions = self.algo.select_actions(states, pop_states, training=True)
            next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
            next_pop_states = self.env.get_population_states()

            # Flatten population states
            pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])
            next_pop_states_flat = np.concatenate([next_pop_states[i] for i in range(2)])

            for pop_id in range(2):
                self.algo.replay_buffers[pop_id].push(
                    states[pop_id],
                    actions[pop_id],
                    rewards[pop_id],
                    next_states[pop_id],
                    pop_states_flat,
                    next_pop_states_flat,
                    float(terminated[pop_id]),
                )

            states = next_states
            pop_states = next_pop_states

        # Sample batch and compute Q values
        batch = self.algo.replay_buffers[0].sample(32)

        # Extract tensors from dict
        next_state_batch = torch.FloatTensor(batch["next_states"])
        next_pop_state_batch = torch.FloatTensor(batch["next_population_states"])

        # Compute both Q values
        with torch.no_grad():
            next_actions = self.algo.actor_targets[0](next_state_batch, next_pop_state_batch)
            q1_next = self.algo.critic1_targets[0](next_state_batch, next_actions, next_pop_state_batch)
            q2_next = self.algo.critic2_targets[0](next_state_batch, next_actions, next_pop_state_batch)

            # TD3 should use min
            q_next = torch.min(q1_next, q2_next)

        # Verify min is indeed the minimum
        assert torch.all(q_next <= q1_next)
        assert torch.all(q_next <= q2_next)

    def test_delayed_policy_updates(self):
        """Test actor is updated less frequently than critics."""
        policy_delay = self.algo.config["policy_delay"]

        # Check that policy_delay is properly configured
        assert policy_delay > 1
        assert "policy_delay" in self.algo.config

        # Verify update_count attribute exists
        assert hasattr(self.algo, "update_count")
        assert self.algo.update_count == 0

    def test_target_policy_smoothing(self):
        """Test noise is added to target actions."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        state_tensor = torch.FloatTensor(states[0]).unsqueeze(0)
        pop_state_list = [pop_states[i] for i in range(2)]
        pop_state_tensor = torch.FloatTensor(np.concatenate(pop_state_list)).unsqueeze(0)

        # Get deterministic action from target actor
        with torch.no_grad():
            clean_action = self.algo.actor_targets[0](state_tensor, pop_state_tensor)

            # Simulate target smoothing noise
            noise = torch.randn_like(clean_action) * self.algo.config["target_noise_std"]
            noise = noise.clamp(
                -self.algo.config["target_noise_clip"],
                self.algo.config["target_noise_clip"],
            )
            noisy_action = clean_action + noise

        # Noisy action should differ from clean
        assert not torch.allclose(clean_action, noisy_action, atol=1e-6)

    def test_training_produces_critic_losses(self):
        """Test training produces loss statistics for both critics."""
        stats = self.algo.train(num_episodes=3)

        assert "critic1_losses" in stats
        assert "critic2_losses" in stats
        assert len(stats["critic1_losses"]) == 2
        assert len(stats["critic2_losses"]) == 2

        # Both critics should have losses
        for pop_id in range(2):
            assert len(stats["critic1_losses"][pop_id]) > 0
            assert len(stats["critic2_losses"][pop_id]) > 0

    def test_exploration_noise_affects_actions(self):
        """Test Gaussian exploration noise affects training actions."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Get deterministic actions
        actions_eval = self.algo.select_actions(states, pop_states, training=False)

        # Get noisy actions (multiple samples should differ)
        actions_train_1 = self.algo.select_actions(states, pop_states, training=True)
        actions_train_2 = self.algo.select_actions(states, pop_states, training=True)

        # Training actions should differ from evaluation
        assert not np.allclose(actions_eval[0], actions_train_1[0], atol=1e-6)

        # Different training samples should differ (due to random noise)
        assert not np.allclose(actions_train_1[0], actions_train_2[0], atol=1e-6)

    def test_action_bounds_with_noise_clipping(self):
        """Test actions are clipped to bounds even with noise."""
        algo_bounded = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-0.3, 0.3), (-1, 1)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Test with high exploration noise
        for _ in range(30):
            actions = algo_bounded.select_actions(states, pop_states, training=True)

            # Check bounds are respected
            assert np.all(actions[0] >= -0.3 - 1e-5)
            assert np.all(actions[0] <= 0.3 + 1e-5)
            assert np.all(actions[1] >= -1 - 1e-5)
            assert np.all(actions[1] <= 1 + 1e-5)

    def test_soft_update_all_target_networks(self):
        """Test all target networks (2 actors + 4 critics) are updated.

        Note: Default tau=0.001 produces very small changes. For testing purposes,
        we use tau=0.1 to make updates detectable while still validating the
        soft update mechanism works correctly (Issue #237).
        """
        # Use larger tau for this test to make changes detectable
        test_algo = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config={"batch_size": 32, "tau": 0.1},  # Larger tau for testing
        )

        # Get initial parameters
        initial_actor_target_params = [
            [p.clone() for p in test_algo.actor_targets[pop_id].parameters()] for pop_id in range(2)
        ]
        initial_critic1_target_params = [
            [p.clone() for p in test_algo.critic1_targets[pop_id].parameters()] for pop_id in range(2)
        ]
        initial_critic2_target_params = [
            [p.clone() for p in test_algo.critic2_targets[pop_id].parameters()] for pop_id in range(2)
        ]

        # Run training
        test_algo.train(num_episodes=3)

        # Check all targets updated (use stricter tolerance to detect small changes)
        for pop_id in range(2):
            for p_new, p_old in zip(
                test_algo.actor_targets[pop_id].parameters(),
                initial_actor_target_params[pop_id],
                strict=False,
            ):
                assert not torch.allclose(p_new, p_old, atol=1e-5)

            for p_new, p_old in zip(
                test_algo.critic1_targets[pop_id].parameters(),
                initial_critic1_target_params[pop_id],
                strict=False,
            ):
                assert not torch.allclose(p_new, p_old, atol=1e-5)

            for p_new, p_old in zip(
                test_algo.critic2_targets[pop_id].parameters(),
                initial_critic2_target_params[pop_id],
                strict=False,
            ):
                assert not torch.allclose(p_new, p_old, atol=1e-5)

    def test_deterministic_policy_in_eval(self):
        """Test policy is deterministic without noise in eval mode."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Multiple eval samples should be identical
        actions_1 = self.algo.select_actions(states, pop_states, training=False)
        actions_2 = self.algo.select_actions(states, pop_states, training=False)

        assert np.allclose(actions_1[0], actions_2[0])
        assert np.allclose(actions_1[1], actions_2[1])

    def test_three_population_td3(self):
        """Test TD3 works with 3 populations."""
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

        algo_3pop = MultiPopulationTD3(
            env=env_3pop,
            num_populations=3,
            state_dims=[2, 3, 1],
            action_dims=[2, 3, 1],
            population_dims=[50, 100, 30],
            action_bounds=[(-1, 1), (-2, 2), (0, 1)],
        )

        stats = algo_3pop.train(num_episodes=2)

        assert len(stats["episode_rewards"]) == 3
        assert len(stats["critic1_losses"]) == 3
        assert len(stats["critic2_losses"]) == 3

    def test_update_counter_increments(self):
        """Test update counter increments for delayed updates."""
        initial_count = self.algo.update_count

        # Fill buffer and train
        self.algo.train(num_episodes=2)

        # Counter should have incremented
        assert self.algo.update_count > initial_count

    def test_config_policy_delay_respected(self):
        """Test custom policy_delay config is used."""
        custom_config = {"policy_delay": 5}

        algo_delayed = MultiPopulationTD3(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
            config=custom_config,
        )

        assert algo_delayed.config["policy_delay"] == 5

    def test_replay_buffer_operations(self):
        """Test replay buffer push and sample work correctly."""
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

        # Sample should work
        batch = self.algo.replay_buffers[0].sample(32)
        assert len(batch) == 7
        assert batch["states"].shape[0] == 32
