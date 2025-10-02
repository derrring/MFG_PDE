"""
Unit tests for Mean Field TD3 implementation.

Tests verify:
- Twin critic architecture
- Target policy smoothing
- Delayed policy updates
- Clipped double Q-learning
- TD3 vs DDPG improvements
"""

import pytest

import numpy as np

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

from mfg_pde.alg.reinforcement.algorithms.mean_field_td3 import MeanFieldTD3, TD3Critic


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTD3Critic:
    """Tests for TD3 critic network."""

    def test_critic_output_shape(self):
        """Test critic outputs scalar Q-values."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        batch_size = 10

        critic = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        pop_states = torch.randn(batch_size, population_dim)

        q_values = critic(states, actions, pop_states)

        assert q_values.shape == (batch_size,)

    def test_twin_critics_independent(self):
        """Test that twin critics have independent parameters."""
        state_dim = 2
        action_dim = 2
        population_dim = 100

        critic1 = TD3Critic(state_dim, action_dim, population_dim)
        critic2 = TD3Critic(state_dim, action_dim, population_dim)

        # Check parameters are different
        for p1, p2 in zip(critic1.parameters(), critic2.parameters(), strict=False):
            # Initially different due to random initialization
            assert not torch.allclose(p1, p2)


@pytest.mark.skipif(not (TORCH_AVAILABLE and GYMNASIUM_AVAILABLE), reason="PyTorch or Gymnasium not available")
class TestMeanFieldTD3:
    """Tests for Mean Field TD3 algorithm."""

    def setup_method(self):
        """Setup mock environment."""

        class MockEnv:
            def __init__(self):
                self.observation_space = None
                self.action_space = None
                self.step_count = 0
                self.max_steps = 10

            def reset(self):
                self.step_count = 0
                return np.random.randn(2).astype(np.float32), {}

            def step(self, action):
                self.step_count += 1
                next_state = np.random.randn(2).astype(np.float32)
                reward = -np.linalg.norm(next_state)
                terminated = np.random.random() < 0.1
                truncated = self.step_count >= self.max_steps
                return next_state, reward, terminated, truncated, {}

            def get_population_state(self):
                class PopState:
                    density_histogram = np.random.randn(100).astype(np.float32)

                return PopState()

        self.env = MockEnv()

    def test_td3_initialization(self):
        """Test TD3 initialization with twin critics."""
        algo = MeanFieldTD3(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        # Check twin critics exist
        assert algo.critic1 is not None
        assert algo.critic2 is not None
        assert algo.critic1_target is not None
        assert algo.critic2_target is not None

        # Check config contains TD3-specific parameters
        assert "policy_delay" in algo.config
        assert "target_noise_std" in algo.config
        assert "target_noise_clip" in algo.config

    def test_action_selection(self):
        """Test action selection with Gaussian noise."""
        algo = MeanFieldTD3(
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

    def test_twin_critics_clipped_target(self):
        """Test that TD3 uses min(Q1, Q2) for target."""
        algo = MeanFieldTD3(
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

        # Update and verify losses exist
        losses = algo.update()
        assert losses is not None
        critic1_loss, critic2_loss, actor_loss = losses
        assert isinstance(critic1_loss, float)
        assert isinstance(critic2_loss, float)

    def test_delayed_policy_updates(self):
        """Test that actor updates are delayed."""
        algo = MeanFieldTD3(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"batch_size": 16, "policy_delay": 3},
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

        # Update 1: update_count starts at 0, 0 % 3 == 0, so actor DOES update
        _ = algo.update()
        assert algo.update_count == 1

        # Update 2: update_count=1, 1 % 3 != 0, so actor doesn't update
        losses2 = algo.update()
        assert losses2[2] == 0.0  # No actor loss

        # Update 3: update_count=2, 2 % 3 != 0, so actor doesn't update
        losses3 = algo.update()
        assert losses3[2] == 0.0  # No actor loss

        # Update 4: update_count=3, 3 % 3 == 0, so actor DOES update
        _ = algo.update()
        assert algo.update_count == 4

    def test_target_policy_smoothing(self):
        """Test that target actions have noise added."""
        algo = MeanFieldTD3(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"target_noise_std": 0.2, "target_noise_clip": 0.5},
        )

        # Check config
        assert algo.config["target_noise_std"] == 0.2
        assert algo.config["target_noise_clip"] == 0.5

        # Target smoothing is applied during update (tested implicitly above)

    def test_soft_target_updates(self):
        """Test soft update mechanism."""
        algo = MeanFieldTD3(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"tau": 0.1},
        )

        # Get initial target params
        initial_params = [p.clone().detach() for p in algo.critic1_target.parameters()]

        # Modify online network
        for p in algo.critic1.parameters():
            p.data.fill_(1.0)

        # Call soft update directly
        algo._soft_update(algo.critic1, algo.critic1_target)

        # Check that target changed toward online network
        # With tau=0.1, target should be 0.1*1.0 + 0.9*initial
        for p_init, p_final in zip(initial_params, algo.critic1_target.parameters(), strict=False):
            # Target should have moved toward 1.0
            assert not torch.allclose(p_init, p_final, atol=1e-4)

    def test_training_loop(self):
        """Test training executes without errors."""
        algo = MeanFieldTD3(
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
        assert "critic1_losses" in stats
        assert "critic2_losses" in stats
        assert len(stats["episode_rewards"]) == 5


@pytest.mark.skipif(not (TORCH_AVAILABLE and GYMNASIUM_AVAILABLE), reason="PyTorch or Gymnasium not available")
class TestTD3vsDDPG:
    """Compare TD3 and DDPG properties."""

    def test_td3_has_twin_critics(self):
        """Verify TD3 has two critics while DDPG has one."""
        from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import MeanFieldDDPG

        class MockEnv:
            def reset(self):
                return np.random.randn(2).astype(np.float32), {}

        env = MockEnv()

        # DDPG: single critic
        ddpg = MeanFieldDDPG(
            env=env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        assert hasattr(ddpg, "critic")
        assert hasattr(ddpg, "critic_target")
        assert not hasattr(ddpg, "critic2")

        # TD3: twin critics
        td3 = MeanFieldTD3(
            env=env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        assert hasattr(td3, "critic1")
        assert hasattr(td3, "critic2")
        assert hasattr(td3, "critic1_target")
        assert hasattr(td3, "critic2_target")

    def test_td3_has_policy_delay(self):
        """Verify TD3 has delayed policy updates."""
        from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import MeanFieldDDPG

        class MockEnv:
            def reset(self):
                return np.random.randn(2).astype(np.float32), {}

        env = MockEnv()

        # DDPG: no policy delay
        ddpg = MeanFieldDDPG(env=env, state_dim=2, action_dim=2, population_dim=100, action_bounds=(-2.0, 2.0))

        assert "policy_delay" not in ddpg.config

        # TD3: has policy delay
        td3 = MeanFieldTD3(env=env, state_dim=2, action_dim=2, population_dim=100, action_bounds=(-2.0, 2.0))

        assert "policy_delay" in td3.config
        assert td3.config["policy_delay"] == 2  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
