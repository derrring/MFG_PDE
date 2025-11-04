"""
Unit tests for Mean Field DDPG implementation.

Tests verify:
- Continuous action generation
- Actor-critic architecture
- Target network updates
- Ornstein-Uhlenbeck noise
- Mean field coupling
"""

import pytest

import numpy as np

pytestmark = pytest.mark.optional_torch

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import (
        DDPGActor,
        DDPGCritic,
        MeanFieldDDPG,
        OrnsteinUhlenbeckNoise,
        ReplayBuffer,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGActor:
    """Tests for DDPG actor network."""

    def test_actor_output_shape(self):
        """Test actor outputs correct action shape."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)
        batch_size = 10

        actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        states = torch.randn(batch_size, state_dim)
        pop_states = torch.randn(batch_size, population_dim)

        actions = actor(states, pop_states)

        assert actions.shape == (batch_size, action_dim)

    def test_actor_action_bounds(self):
        """Test actor respects action bounds."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-1.5, 1.5)

        actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        # Test with various inputs
        for _ in range(10):
            states = torch.randn(50, state_dim)
            pop_states = torch.randn(50, population_dim)
            actions = actor(states, pop_states)

            # Check all actions within bounds
            assert torch.all(actions >= action_bounds[0])
            assert torch.all(actions <= action_bounds[1])

    def test_actor_deterministic(self):
        """Test actor produces deterministic output."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)

        actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        state = torch.randn(1, state_dim)
        pop_state = torch.randn(1, population_dim)

        # Same input should give same output
        action1 = actor(state, pop_state)
        action2 = actor(state, pop_state)

        assert torch.allclose(action1, action2)

    def test_actor_population_sensitivity(self):
        """Test actor is sensitive to population state."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)

        actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        state = torch.randn(1, state_dim)
        pop_state1 = torch.randn(1, population_dim)
        pop_state2 = torch.randn(1, population_dim)

        action1 = actor(state, pop_state1)
        action2 = actor(state, pop_state2)

        # Different population should give different actions
        assert not torch.allclose(action1, action2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGCritic:
    """Tests for DDPG critic network."""

    def test_critic_output_shape(self):
        """Test critic outputs scalar Q-values."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        batch_size = 10

        critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        pop_states = torch.randn(batch_size, population_dim)

        q_values = critic(states, actions, pop_states)

        assert q_values.shape == (batch_size,)

    def test_critic_action_dependency(self):
        """Test critic Q-values depend on action."""
        state_dim = 2
        action_dim = 2
        population_dim = 100

        critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        state = torch.randn(1, state_dim)
        pop_state = torch.randn(1, population_dim)
        action1 = torch.randn(1, action_dim)
        action2 = torch.randn(1, action_dim)

        q1 = critic(state, action1, pop_state)
        q2 = critic(state, action2, pop_state)

        # Different actions should give different Q-values
        assert not torch.allclose(q1, q2)

    def test_critic_gradient_flow(self):
        """Test gradients flow through critic."""
        state_dim = 2
        action_dim = 2
        population_dim = 100

        critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        states = torch.randn(10, state_dim, requires_grad=True)
        actions = torch.randn(10, action_dim, requires_grad=True)
        pop_states = torch.randn(10, population_dim)

        q_values = critic(states, actions, pop_states)
        loss = q_values.sum()
        loss.backward()

        # Check gradients exist
        assert states.grad is not None
        assert actions.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOrnsteinUhlenbeckNoise:
    """Tests for OU noise process."""

    def test_noise_shape(self):
        """Test noise has correct shape."""
        action_dim = 3
        noise = OrnsteinUhlenbeckNoise(action_dim=action_dim)

        sample = noise.sample()
        assert sample.shape == (action_dim,)

    def test_noise_reset(self):
        """Test noise reset to initial state."""
        action_dim = 2
        noise = OrnsteinUhlenbeckNoise(action_dim=action_dim, mu=0.5)

        # Sample to change state
        for _ in range(10):
            noise.sample()

        # Reset
        noise.reset()

        # State should be at mu
        assert np.allclose(noise.state, 0.5)

    def test_noise_temporal_correlation(self):
        """Test OU noise has temporal correlation."""
        action_dim = 1
        noise = OrnsteinUhlenbeckNoise(action_dim=action_dim, theta=0.15, sigma=0.2)

        samples = [noise.sample()[0] for _ in range(100)]

        # OU process should have autocorrelation
        # Compute lag-1 autocorrelation
        samples_array = np.array(samples)
        mean = samples_array.mean()
        var = samples_array.var()

        if var > 0:
            autocorr = np.corrcoef(samples_array[:-1] - mean, samples_array[1:] - mean)[0, 1]
            # OU process should have positive autocorrelation
            assert autocorr > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestReplayBuffer:
    """Tests for experience replay buffer."""

    def test_buffer_push_and_sample(self):
        """Test pushing and sampling from buffer."""
        capacity = 100
        state_dim = 2
        action_dim = 2
        pop_dim = 50

        buffer = ReplayBuffer(capacity, state_dim, action_dim, pop_dim)

        # Push some transitions
        for i in range(50):
            buffer.push(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=float(i),
                next_state=np.random.randn(state_dim),
                population_state=np.random.randn(pop_dim),
                next_population_state=np.random.randn(pop_dim),
                done=False,
            )

        assert len(buffer) == 50

        # Sample batch
        batch = buffer.sample(batch_size=32)
        assert batch["states"].shape == (32, state_dim)
        assert batch["actions"].shape == (32, action_dim)
        assert batch["rewards"].shape == (32,)

    def test_buffer_capacity(self):
        """Test buffer respects capacity limit."""
        capacity = 10
        buffer = ReplayBuffer(capacity, state_dim=2, action_dim=2, pop_dim=10)

        # Push more than capacity
        for _ in range(20):
            buffer.push(
                state=np.random.randn(2),
                action=np.random.randn(2),
                reward=0.0,
                next_state=np.random.randn(2),
                population_state=np.random.randn(10),
                next_population_state=np.random.randn(10),
                done=False,
            )

        # Size should be capped at capacity
        assert len(buffer) == capacity


@pytest.mark.skipif(not (TORCH_AVAILABLE and GYMNASIUM_AVAILABLE), reason="PyTorch or Gymnasium not available")
class TestMeanFieldDDPG:
    """Tests for Mean Field DDPG algorithm."""

    def setup_method(self):
        """Setup mock environment."""

        class MockEnv:
            def __init__(self):
                self.observation_space = None
                self.action_space = None
                self.step_count = 0
                self.max_steps = 10  # Short episodes for testing

            def reset(self):
                self.step_count = 0
                return np.random.randn(2).astype(np.float32), {}

            def step(self, action):
                self.step_count += 1
                next_state = np.random.randn(2).astype(np.float32)
                reward = -np.linalg.norm(next_state)
                terminated = np.random.random() < 0.1  # 10% chance to terminate
                truncated = self.step_count >= self.max_steps
                return next_state, reward, terminated, truncated, {}

            def get_population_state(self):
                class PopState:
                    density_histogram = np.random.randn(100).astype(np.float32)

                return PopState()

        self.env = MockEnv()

    def test_ddpg_initialization(self):
        """Test DDPG initialization."""
        algo = MeanFieldDDPG(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        assert algo.state_dim == 2
        assert algo.action_dim == 2
        assert algo.population_dim == 100

    def test_action_selection(self):
        """Test action selection."""
        algo = MeanFieldDDPG(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        state = np.random.randn(2).astype(np.float32)
        pop_state = np.random.randn(100).astype(np.float32)

        # Test with exploration
        action_train = algo.select_action(state, pop_state, training=True)
        assert action_train.shape == (2,)
        assert np.all(action_train >= -2.0)
        assert np.all(action_train <= 2.0)

        # Test without exploration
        action_eval = algo.select_action(state, pop_state, training=False)
        assert action_eval.shape == (2,)

    def test_update_mechanism(self):
        """Test network update."""
        algo = MeanFieldDDPG(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"batch_size": 32},
        )

        # Add experiences
        for _ in range(50):
            algo.replay_buffer.push(
                state=np.random.randn(2).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                reward=np.random.randn(),
                next_state=np.random.randn(2).astype(np.float32),
                population_state=np.random.randn(100).astype(np.float32),
                next_population_state=np.random.randn(100).astype(np.float32),
                done=False,
            )

        # Update
        losses = algo.update()
        assert losses is not None
        critic_loss, actor_loss = losses
        assert isinstance(critic_loss, float)
        assert isinstance(actor_loss, float)

    def test_target_network_update(self):
        """Test soft update of target networks."""
        algo = MeanFieldDDPG(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"tau": 0.01},
        )

        # Get initial target params
        initial_actor_params = [p.clone() for p in algo.actor_target.parameters()]

        # Update online network
        for p in algo.actor.parameters():
            p.data += 0.1

        # Soft update
        algo._soft_update(algo.actor, algo.actor_target)

        # Target should have changed slightly
        for p_old, p_new in zip(initial_actor_params, algo.actor_target.parameters(), strict=False):
            assert not torch.allclose(p_old, p_new)

    def test_training_loop(self):
        """Test training executes without errors."""
        algo = MeanFieldDDPG(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"batch_size": 16},
        )

        # Short training run
        stats = algo.train(num_episodes=5)

        assert "episode_rewards" in stats
        assert "episode_lengths" in stats
        assert len(stats["episode_rewards"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
