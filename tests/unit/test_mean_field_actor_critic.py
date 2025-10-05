"""
Tests for Mean Field Actor-Critic algorithm.

Tests cover:
- Network initialization
- Action selection
- Training workflow
- GAE computation
"""

import pytest

import numpy as np

torch = pytest.importorskip("torch")

from mfg_pde.alg.reinforcement.algorithms import (  # noqa: E402
    ActorNetwork,
    CriticNetwork,
    MeanFieldActorCritic,
)


class TestActorNetwork:
    """Test Actor Network."""

    def test_actor_initialization(self):
        """Test actor network initialization."""
        state_dim = 4
        action_dim = 4
        population_dim = 10

        actor = ActorNetwork(state_dim, action_dim, population_dim)

        assert actor.state_dim == state_dim
        assert actor.action_dim == action_dim
        assert actor.population_dim == population_dim

    def test_actor_forward_pass(self):
        """Test forward pass through actor."""
        batch_size = 8
        state_dim = 4
        action_dim = 4
        population_dim = 10

        actor = ActorNetwork(state_dim, action_dim, population_dim)

        state = torch.randn(batch_size, state_dim)
        population = torch.randn(batch_size, population_dim)

        logits = actor(state, population)

        assert logits.shape == (batch_size, action_dim)

    def test_actor_get_action(self):
        """Test action sampling."""
        state_dim = 4
        action_dim = 4
        population_dim = 10

        actor = ActorNetwork(state_dim, action_dim, population_dim)

        state = torch.randn(1, state_dim)
        population = torch.randn(1, population_dim)

        # Stochastic action
        action, log_prob = actor.get_action(state, population, deterministic=False)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < action_dim

        # Deterministic action
        action_det, _log_prob_det = actor.get_action(state, population, deterministic=True)
        assert action_det.shape == (1,)
        assert 0 <= action_det.item() < action_dim


class TestCriticNetwork:
    """Test Critic Network."""

    def test_critic_v_initialization(self):
        """Test V-critic initialization."""
        state_dim = 4
        population_dim = 10

        critic = CriticNetwork(state_dim, population_dim, critic_type="v")

        assert critic.state_dim == state_dim
        assert critic.population_dim == population_dim
        assert critic.critic_type == "v"

    def test_critic_q_initialization(self):
        """Test Q-critic initialization."""
        state_dim = 4
        action_dim = 4
        population_dim = 10

        critic = CriticNetwork(state_dim, population_dim, action_dim=action_dim, critic_type="q")

        assert critic.state_dim == state_dim
        assert critic.action_dim == action_dim
        assert critic.critic_type == "q"

    def test_critic_v_forward_pass(self):
        """Test V-critic forward pass."""
        batch_size = 8
        state_dim = 4
        population_dim = 10

        critic = CriticNetwork(state_dim, population_dim, critic_type="v")

        state = torch.randn(batch_size, state_dim)
        population = torch.randn(batch_size, population_dim)

        value = critic(state, population)

        assert value.shape == (batch_size, 1)

    def test_critic_q_forward_pass(self):
        """Test Q-critic forward pass."""
        batch_size = 8
        state_dim = 4
        action_dim = 4
        population_dim = 10

        critic = CriticNetwork(state_dim, population_dim, action_dim=action_dim, critic_type="q")

        state = torch.randn(batch_size, state_dim)
        population = torch.randn(batch_size, population_dim)
        action = torch.randint(0, action_dim, (batch_size,))

        value = critic(state, population, action)

        assert value.shape == (batch_size, 1)


class MockMFGEnv:
    """Mock MFG environment for testing."""

    def __init__(self, state_dim=4, action_dim=4, population_dim=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim
        self.steps = 0
        self.max_steps = 10

    def reset(self):
        """Reset environment."""
        self.steps = 0
        return {
            "agent_position": np.random.randn(self.state_dim),
            "local_density": np.random.rand(self.population_dim),
        }

    def step(self, action):
        """Take step in environment."""
        self.steps += 1
        done = self.steps >= self.max_steps
        reward = np.random.randn()

        obs = {
            "agent_position": np.random.randn(self.state_dim),
            "local_density": np.random.rand(self.population_dim),
        }

        return obs, reward, done, False, {}


class TestMeanFieldActorCritic:
    """Test Mean Field Actor-Critic algorithm."""

    def test_mfac_initialization(self):
        """Test MFAC initialization."""
        env = MockMFGEnv()
        mfac = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")

        assert mfac.state_dim == 4
        assert mfac.action_dim == 4
        assert mfac.population_dim == 10
        assert mfac.device.type == "cpu"

    def test_select_action(self):
        """Test action selection."""
        env = MockMFGEnv()
        mfac = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")

        state = np.random.randn(4)
        population = np.random.rand(10)

        action, log_prob = mfac.select_action(state, population)

        assert isinstance(action, int)
        assert 0 <= action < 4
        assert isinstance(log_prob, float)

    def test_compute_gae(self):
        """Test GAE computation."""
        env = MockMFGEnv()
        mfac = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")

        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        next_value = 2.0
        dones = [False, False, True]

        advantages, returns = mfac.compute_gae(rewards, values, next_value, dones)

        assert len(advantages) == 3
        assert len(returns) == 3
        assert all(isinstance(a, float) for a in advantages)
        assert all(isinstance(r, float) for r in returns)

    def test_training_smoke_test(self):
        """Smoke test for training loop."""
        env = MockMFGEnv()
        mfac = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")

        # Train for just a few episodes as smoke test
        stats = mfac.train(num_episodes=2, max_steps_per_episode=5, log_interval=1)

        assert "episode_rewards" in stats
        assert "episode_lengths" in stats
        assert len(stats["episode_rewards"]) == 2
        assert len(stats["episode_lengths"]) == 2

    def test_save_load(self, tmp_path):
        """Test model save/load."""
        env = MockMFGEnv()
        mfac = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")

        # Save model
        save_path = tmp_path / "mfac_checkpoint.pt"
        mfac.save(str(save_path))

        assert save_path.exists()

        # Create new agent and load
        mfac2 = MeanFieldActorCritic(env, state_dim=4, action_dim=4, population_dim=10, device="cpu")
        mfac2.load(str(save_path))

        # Check loaded correctly
        assert mfac2.training_step == mfac.training_step
