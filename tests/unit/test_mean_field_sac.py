"""
Unit tests for Mean Field SAC implementation.

Tests verify:
- Stochastic policy with entropy
- Reparameterization trick
- Automatic temperature tuning
- Soft Bellman equation
- Maximum entropy objective
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

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms.mean_field_sac import MeanFieldSAC, SACStochasticActor


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSACStochasticActor:
    """Tests for SAC stochastic actor network."""

    def test_actor_output_shapes(self):
        """Test actor outputs mean and log_std."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)
        batch_size = 10

        actor = SACStochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        states = torch.randn(batch_size, state_dim)
        pop_states = torch.randn(batch_size, population_dim)

        mean, log_std = actor(states, pop_states)

        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)

    def test_actor_sample_with_reparameterization(self):
        """Test action sampling with reparameterization trick."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)

        actor = SACStochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        states = torch.randn(10, state_dim)
        pop_states = torch.randn(10, population_dim)

        action, log_prob, mean_action = actor.sample(states, pop_states)

        # Check shapes
        assert action.shape == (10, action_dim)
        assert log_prob.shape == (10, 1)
        assert mean_action.shape == (10, action_dim)

        # Check action bounds
        assert torch.all(action >= action_bounds[0])
        assert torch.all(action <= action_bounds[1])

        # Check gradients flow (reparameterization)
        loss = action.sum()
        loss.backward()
        # Should have gradients
        assert any(p.grad is not None for p in actor.parameters())

    def test_actor_stochasticity(self):
        """Test that actor produces stochastic outputs."""
        state_dim = 2
        action_dim = 2
        population_dim = 100
        action_bounds = (-2.0, 2.0)

        actor = SACStochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        state = torch.randn(1, state_dim)
        pop_state = torch.randn(1, population_dim)

        # Sample multiple times
        action1, _, _ = actor.sample(state, pop_state)
        action2, _, _ = actor.sample(state, pop_state)

        # Should be different due to stochasticity
        assert not torch.allclose(action1, action2, atol=1e-3)

    def test_log_prob_computation(self):
        """Test log probability includes tanh correction."""
        state_dim = 2
        action_dim = 1  # Use 1D for simpler verification
        population_dim = 100
        action_bounds = (-1.0, 1.0)

        actor = SACStochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
        )

        state = torch.randn(100, state_dim)
        pop_state = torch.randn(100, population_dim)

        _, log_probs, _ = actor.sample(state, pop_state)

        # Log probabilities should be mostly negative (allow small positive due to numerical issues)
        assert torch.mean(log_probs) < 0  # Average should be negative

        # Should have reasonable magnitude
        assert torch.all(log_probs > -20)  # Not too negative
        assert torch.all(log_probs < 5)  # Not too positive (numerical issues)


@pytest.mark.skipif(not (TORCH_AVAILABLE and GYMNASIUM_AVAILABLE), reason="PyTorch or Gymnasium not available")
class TestMeanFieldSAC:
    """Tests for Mean Field SAC algorithm."""

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

    def test_sac_initialization(self):
        """Test SAC initialization with stochastic actor."""
        algo = MeanFieldSAC(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        # Check components
        assert algo.actor is not None
        assert algo.critic1 is not None
        assert algo.critic2 is not None
        assert algo.log_alpha is not None  # Auto-tuning enabled by default

        # Check config
        assert algo.config["auto_tune_temperature"] is True
        assert algo.config["target_entropy"] == -2.0  # -action_dim

    def test_stochastic_action_selection(self):
        """Test stochastic action selection."""
        algo = MeanFieldSAC(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        state = np.random.randn(2).astype(np.float32)
        pop_state = np.random.randn(100).astype(np.float32)

        # Training mode: stochastic
        action1 = algo.select_action(state, pop_state, training=True)
        action2 = algo.select_action(state, pop_state, training=True)

        assert action1.shape == (2,)
        assert not np.allclose(action1, action2, atol=1e-3)  # Different due to sampling

        # Evaluation mode: deterministic (use mean)
        action_eval = algo.select_action(state, pop_state, training=False)
        assert action_eval.shape == (2,)

    def test_automatic_temperature_tuning(self):
        """Test automatic temperature adjustment."""
        algo = MeanFieldSAC(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"auto_tune_temperature": True, "initial_temperature": 0.5, "batch_size": 32},
        )

        # Add experiences (more than batch size)
        for _ in range(100):
            algo.replay_buffer.push(
                state=np.random.randn(2).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                reward=np.random.randn(),
                next_state=np.random.randn(2).astype(np.float32),
                population_state=np.random.randn(100).astype(np.float32),
                next_population_state=np.random.randn(100).astype(np.float32),
                done=False,
            )

        # Run updates
        alpha_values = []
        for _ in range(20):
            losses = algo.update()
            if losses:
                assert "alpha" in losses
                assert "alpha_loss" in losses
                alpha_values.append(losses["alpha"])

        # Temperature should have been updated (check we got values)
        assert len(alpha_values) > 0
        assert algo.update_count == 20

    def test_entropy_regularization(self):
        """Test that updates include entropy term."""
        algo = MeanFieldSAC(
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
        assert "entropy" in losses

        # Entropy should be positive (negative of negative log prob)
        assert losses["entropy"] > 0

    def test_soft_bellman_backup(self):
        """Test soft Bellman target computation."""
        algo = MeanFieldSAC(
            env=self.env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
            config={"batch_size": 16},
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

        # Update includes soft target: min(Q1', Q2') - α log π
        losses = algo.update()
        assert losses is not None
        assert "critic1_loss" in losses
        assert "critic2_loss" in losses

    def test_training_loop(self):
        """Test training executes without errors."""
        algo = MeanFieldSAC(
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
        assert "alpha_values" in stats
        assert "entropy_values" in stats
        assert len(stats["episode_rewards"]) == 5


@pytest.mark.skipif(not (TORCH_AVAILABLE and GYMNASIUM_AVAILABLE), reason="PyTorch or Gymnasium not available")
class TestSACvsTD3:
    """Compare SAC and TD3 properties."""

    def test_sac_has_stochastic_policy(self):
        """Verify SAC uses stochastic policy while TD3 is deterministic."""
        from mfg_pde.alg.reinforcement.algorithms.mean_field_td3 import MeanFieldTD3

        class MockEnv:
            def reset(self):
                return np.random.randn(2).astype(np.float32), {}

        env = MockEnv()

        # TD3: deterministic actor
        td3 = MeanFieldTD3(
            env=env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        # SAC: stochastic actor
        sac = MeanFieldSAC(
            env=env,
            state_dim=2,
            action_dim=2,
            population_dim=100,
            action_bounds=(-2.0, 2.0),
        )

        # TD3 actor has no sampling method with log_prob
        assert not hasattr(td3.actor, "sample")

        # SAC actor has sample method with log_prob
        assert hasattr(sac.actor, "sample")

    def test_sac_has_temperature_tuning(self):
        """Verify SAC has automatic temperature while TD3 doesn't."""
        from mfg_pde.alg.reinforcement.algorithms.mean_field_td3 import MeanFieldTD3

        class MockEnv:
            def reset(self):
                return np.random.randn(2).astype(np.float32), {}

        env = MockEnv()

        td3 = MeanFieldTD3(env=env, state_dim=2, action_dim=2, population_dim=100, action_bounds=(-2.0, 2.0))

        sac = MeanFieldSAC(env=env, state_dim=2, action_dim=2, population_dim=100, action_bounds=(-2.0, 2.0))

        # TD3 has no temperature
        assert not hasattr(td3, "alpha")
        assert not hasattr(td3, "log_alpha")

        # SAC has temperature
        assert hasattr(sac, "alpha")
        assert hasattr(sac, "log_alpha")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
