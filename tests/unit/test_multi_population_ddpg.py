"""
Unit tests for Multi-Population DDPG algorithm.

Tests training loop, replay buffer operations, network updates, and convergence behavior.
"""

from __future__ import annotations

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

from mfg_pde.alg.reinforcement.algorithms import MultiPopulationDDPG
from tests.unit.test_multi_population_env import SimpleMultiPopEnv


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiPopulationDDPG:
    """Comprehensive tests for MultiPopulationDDPG."""

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

        self.algo = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1)],
        )

    def test_replay_buffer_store_and_sample(self):
        """Test replay buffer can store and sample transitions."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        actions = self.algo.select_actions(states, pop_states, training=True)
        next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
        next_pop_states = self.env.get_population_states()

        # Flatten population states for storage
        pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])
        next_pop_states_flat = np.concatenate([next_pop_states[i] for i in range(2)])

        # Store transitions for each population
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

        # Check buffer has data
        assert len(self.algo.replay_buffers[0]) == 1
        assert len(self.algo.replay_buffers[1]) == 1

        # Fill buffer to enable sampling
        for _ in range(self.algo.config["batch_size"]):
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

        # Test sampling
        batch = self.algo.replay_buffers[0].sample(32)
        assert len(batch) == 7
        assert batch["states"].shape == (32, 2)  # states

    def test_training_episode(self):
        """Test a single training episode completes."""
        initial_buffer_sizes = [len(buf) for buf in self.algo.replay_buffers]

        stats = self.algo.train(num_episodes=1)

        # Check stats structure
        assert "episode_rewards" in stats
        assert "episode_lengths" in stats
        assert len(stats["episode_rewards"]) == 2
        assert len(stats["episode_lengths"]) == 1

        # Check replay buffers grew
        for pop_id in range(2):
            assert len(self.algo.replay_buffers[pop_id]) > initial_buffer_sizes[pop_id]

    def test_multiple_episodes_training(self):
        """Test training over multiple episodes."""
        num_episodes = 5
        stats = self.algo.train(num_episodes=num_episodes)

        # Check all episodes recorded
        assert len(stats["episode_lengths"]) == num_episodes
        for pop_id in range(2):
            assert len(stats["episode_rewards"][pop_id]) == num_episodes

        # Check rewards are floats
        for pop_id in range(2):
            for reward in stats["episode_rewards"][pop_id]:
                assert isinstance(reward, float)

    def test_ou_noise_affects_training_actions(self):
        """Test OU noise is added during training mode."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Get deterministic actions (evaluation)
        actions_eval = self.algo.select_actions(states, pop_states, training=False)

        # Get noisy actions (training)
        actions_train = self.algo.select_actions(states, pop_states, training=True)

        # Actions should differ due to noise
        assert not np.allclose(actions_eval[0], actions_train[0], atol=1e-6)
        assert not np.allclose(actions_eval[1], actions_train[1], atol=1e-6)

    def test_soft_update_target_networks(self):
        """Test target networks are soft-updated."""
        # Fill replay buffer first
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()
        pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])

        for _ in range(self.algo.config["batch_size"] + 50):
            actions = self.algo.select_actions(states, pop_states, training=True)
            next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
            next_pop_states = self.env.get_population_states()
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
            pop_states_flat = next_pop_states_flat

        # Get initial target parameters
        initial_actor_params = [p.clone() for p in self.algo.actor_targets[0].parameters()]
        initial_critic_params = [p.clone() for p in self.algo.critic_targets[0].parameters()]

        # Run training to trigger updates
        self.algo.train(num_episodes=5)

        # Check target parameters changed (soft update)
        for p_new, p_old in zip(self.algo.actor_targets[0].parameters(), initial_actor_params, strict=False):
            assert not torch.allclose(p_new, p_old, atol=1e-5)

        for p_new, p_old in zip(self.algo.critic_targets[0].parameters(), initial_critic_params, strict=False):
            assert not torch.allclose(p_new, p_old, atol=1e-5)

    def test_critic_loss_computation(self):
        """Test critic loss is computed correctly."""
        # Fill replay buffer
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()
        pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])

        for _ in range(self.algo.config["batch_size"] + 10):
            actions = self.algo.select_actions(states, pop_states, training=True)
            next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
            next_pop_states = self.env.get_population_states()
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
            pop_states_flat = next_pop_states_flat

        # Perform update and check loss is computed
        stats = self.algo.train(num_episodes=1)

        assert "critic_losses" in stats
        assert len(stats["critic_losses"]) == 2
        assert len(stats["critic_losses"][0]) > 0

    def test_action_bounds_maintained_during_training(self):
        """Test actions stay within bounds during training."""
        algo_bounded = MultiPopulationDDPG(
            env=self.env,
            num_populations=2,
            state_dims=2,
            action_dims=[2, 2],
            population_dims=50,
            action_bounds=[(-0.5, 0.5), (-1, 1)],
        )

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Test actions over multiple samples
        for _ in range(20):
            actions = algo_bounded.select_actions(states, pop_states, training=True)

            # Check pop 0 bounds [-0.5, 0.5]
            assert np.all(actions[0] >= -0.5 - 1e-5)
            assert np.all(actions[0] <= 0.5 + 1e-5)

            # Check pop 1 bounds [-1, 1]
            assert np.all(actions[1] >= -1 - 1e-5)
            assert np.all(actions[1] <= 1 + 1e-5)

    def test_cross_population_awareness_in_critic(self):
        """Test critic observes all population states."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()
        actions = self.algo.select_actions(states, pop_states, training=False)

        # Critic should take all population states
        state_tensor = torch.FloatTensor(states[0]).unsqueeze(0)
        action_tensor = torch.FloatTensor(actions[0]).unsqueeze(0)

        # Concatenate all population states
        pop_state_list = [pop_states[i] for i in range(2)]
        pop_state_tensor = torch.FloatTensor(np.concatenate(pop_state_list)).unsqueeze(0)

        # Critic forward pass should work
        q_value = self.algo.critics[0](state_tensor, action_tensor, pop_state_tensor)
        assert q_value.shape == (1,) or q_value.shape == (1, 1)

    def test_deterministic_policy_evaluation(self):
        """Test policy is deterministic in evaluation mode."""
        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()

        # Get multiple action samples in eval mode
        actions_1 = self.algo.select_actions(states, pop_states, training=False)
        actions_2 = self.algo.select_actions(states, pop_states, training=False)

        # Should be identical (deterministic)
        assert np.allclose(actions_1[0], actions_2[0])
        assert np.allclose(actions_1[1], actions_2[1])

    def test_replay_buffer_respects_capacity(self):
        """Test replay buffer doesn't exceed capacity."""
        buffer_capacity = self.algo.config["replay_buffer_size"]

        states, _ = self.env.reset(seed=42)
        pop_states = self.env.get_population_states()
        actions = self.algo.select_actions(states, pop_states, training=True)
        next_states, rewards, terminated, _truncated, _ = self.env.step(actions)
        next_pop_states = self.env.get_population_states()

        # Flatten population states
        pop_states_flat = np.concatenate([pop_states[i] for i in range(2)])
        next_pop_states_flat = np.concatenate([next_pop_states[i] for i in range(2)])

        # Overfill buffer
        for _ in range(buffer_capacity + 100):
            self.algo.replay_buffers[0].push(
                states[0],
                actions[0],
                rewards[0],
                next_states[0],
                pop_states_flat,
                next_pop_states_flat,
                float(terminated[0]),
            )

        # Should not exceed capacity
        assert len(self.algo.replay_buffers[0]) <= buffer_capacity

    def test_three_population_training(self):
        """Test DDPG works with 3 populations."""
        env_3pop = SimpleMultiPopEnv(
            num_populations=3,
            state_dims=[2, 2, 2],
            action_specs=[
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
                {"type": "continuous", "dim": 1, "bounds": (-1, 1)},
            ],
            population_sizes=50,
            max_steps=10,
        )

        algo_3pop = MultiPopulationDDPG(
            env=env_3pop,
            num_populations=3,
            state_dims=[2, 2, 2],
            action_dims=[2, 2, 1],
            population_dims=50,
            action_bounds=[(-1, 1), (-1, 1), (-1, 1)],
        )

        stats = algo_3pop.train(num_episodes=2)

        assert len(stats["episode_rewards"]) == 3
        assert len(stats["episode_lengths"]) == 2
