"""
Unit tests for multi-population neural networks.

Tests JointPopulationEncoder, MultiPopulationActor, MultiPopulationCritic,
and MultiPopulationStochasticActor.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import pytest

# Skip tests if PyTorch not available
torch = pytest.importorskip("torch")

from mfg_pde.alg.reinforcement.multi_population.networks import (  # noqa: E402
    JointPopulationEncoder,
    MultiPopulationActor,
    MultiPopulationCritic,
    MultiPopulationStochasticActor,
)
from mfg_pde.alg.reinforcement.multi_population.population_config import (  # noqa: E402
    PopulationConfig,
)


@pytest.fixture
def test_populations():
    """Create test population configurations."""
    return {
        "pop1": PopulationConfig(
            population_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            algorithm="ddpg",
        ),
        "pop2": PopulationConfig(
            population_id="pop2",
            state_dim=3,
            action_dim=2,
            action_bounds=(-2.0, 2.0),
            algorithm="td3",
        ),
    }


class TestJointPopulationEncoder:
    """Tests for JointPopulationEncoder."""

    def test_initialization(self, test_populations):
        """Test encoder initialization."""
        encoder = JointPopulationEncoder(
            population_configs=test_populations,
            hidden_dim=64,
            use_attention=False,
        )

        assert encoder.output_dim == 32  # hidden_dim // 2
        assert len(encoder.pop_encoders) == 2
        assert "pop1" in encoder.pop_encoders
        assert "pop2" in encoder.pop_encoders

    def test_forward_no_attention(self, test_populations):
        """Test forward pass without attention."""
        encoder = JointPopulationEncoder(
            population_configs=test_populations,
            hidden_dim=64,
            use_attention=False,
        )

        batch_size = 8
        pop_states = {
            "pop1": torch.randn(batch_size, 20),  # 2 * 10 bins
            "pop2": torch.randn(batch_size, 30),  # 3 * 10 bins
        }

        output = encoder(pop_states)

        assert output.shape == (batch_size, 32)
        assert not torch.isnan(output).any()

    def test_forward_with_attention(self, test_populations):
        """Test forward pass with attention."""
        encoder = JointPopulationEncoder(
            population_configs=test_populations,
            hidden_dim=64,
            use_attention=True,
        )

        batch_size = 8
        pop_states = {
            "pop1": torch.randn(batch_size, 20),
            "pop2": torch.randn(batch_size, 30),
        }

        output = encoder(pop_states)

        assert output.shape == (batch_size, 32)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, test_populations):
        """Test that gradients flow through encoder."""
        encoder = JointPopulationEncoder(
            population_configs=test_populations,
            hidden_dim=64,
        )

        pop_states = {
            "pop1": torch.randn(4, 20, requires_grad=True),
            "pop2": torch.randn(4, 30, requires_grad=True),
        }

        output = encoder(pop_states)
        loss = output.sum()
        loss.backward()

        assert pop_states["pop1"].grad is not None
        assert pop_states["pop2"].grad is not None


class TestMultiPopulationActor:
    """Tests for MultiPopulationActor."""

    def test_initialization(self, test_populations):
        """Test actor initialization."""
        actor = MultiPopulationActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
            hidden_dims=[128, 64],
        )

        assert actor.pop_id == "pop1"
        assert actor.action_bounds == (-1.0, 1.0)

    def test_forward(self, test_populations):
        """Test actor forward pass."""
        actor = MultiPopulationActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        batch_size = 8
        state = torch.randn(batch_size, 2)
        pop_states = {
            "pop1": torch.randn(batch_size, 20),
            "pop2": torch.randn(batch_size, 30),
        }

        action = actor(state, pop_states)

        assert action.shape == (batch_size, 1)
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)

    def test_action_bounds(self, test_populations):
        """Test that actions respect bounds."""
        actor = MultiPopulationActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-2.5, 1.5),
            population_configs=test_populations,
        )

        state = torch.randn(16, 2)
        pop_states = {
            "pop1": torch.randn(16, 20),
            "pop2": torch.randn(16, 30),
        }

        action = actor(state, pop_states)

        assert torch.all(action >= -2.5)
        assert torch.all(action <= 1.5)

    def test_gradient_flow(self, test_populations):
        """Test gradient flow through actor."""
        actor = MultiPopulationActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        state = torch.randn(4, 2, requires_grad=True)
        pop_states = {
            "pop1": torch.randn(4, 20),
            "pop2": torch.randn(4, 30),
        }

        action = actor(state, pop_states)
        loss = action.sum()
        loss.backward()

        assert state.grad is not None


class TestMultiPopulationCritic:
    """Tests for MultiPopulationCritic."""

    def test_initialization(self, test_populations):
        """Test critic initialization."""
        critic = MultiPopulationCritic(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            population_configs=test_populations,
        )

        assert critic.pop_id == "pop1"

    def test_forward(self, test_populations):
        """Test critic forward pass."""
        critic = MultiPopulationCritic(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            population_configs=test_populations,
        )

        batch_size = 8
        state = torch.randn(batch_size, 2)
        action = torch.randn(batch_size, 1)
        pop_states = {
            "pop1": torch.randn(batch_size, 20),
            "pop2": torch.randn(batch_size, 30),
        }

        q_value = critic(state, action, pop_states)

        assert q_value.shape == (batch_size,)
        assert not torch.isnan(q_value).any()

    def test_gradient_flow(self, test_populations):
        """Test gradient flow through critic."""
        critic = MultiPopulationCritic(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            population_configs=test_populations,
        )

        state = torch.randn(4, 2, requires_grad=True)
        action = torch.randn(4, 1, requires_grad=True)
        pop_states = {
            "pop1": torch.randn(4, 20),
            "pop2": torch.randn(4, 30),
        }

        q_value = critic(state, action, pop_states)
        loss = q_value.sum()
        loss.backward()

        assert state.grad is not None
        assert action.grad is not None


class TestMultiPopulationStochasticActor:
    """Tests for MultiPopulationStochasticActor (SAC)."""

    def test_initialization(self, test_populations):
        """Test stochastic actor initialization."""
        actor = MultiPopulationStochasticActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        assert actor.pop_id == "pop1"

    def test_forward_returns_mean_log_std(self, test_populations):
        """Test that forward returns mean and log_std."""
        actor = MultiPopulationStochasticActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        state = torch.randn(8, 2)
        pop_states = {
            "pop1": torch.randn(8, 20),
            "pop2": torch.randn(8, 30),
        }

        mean, log_std = actor(state, pop_states)

        assert mean.shape == (8, 1)
        assert log_std.shape == (8, 1)
        assert torch.all(log_std >= -20)  # LOG_STD_MIN
        assert torch.all(log_std <= 2)  # LOG_STD_MAX

    def test_sample(self, test_populations):
        """Test action sampling with reparameterization."""
        actor = MultiPopulationStochasticActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        state = torch.randn(8, 2)
        pop_states = {
            "pop1": torch.randn(8, 20),
            "pop2": torch.randn(8, 30),
        }

        action, log_prob, mean_action = actor.sample(state, pop_states)

        assert action.shape == (8, 1)
        assert log_prob.shape == (8, 1)
        assert mean_action.shape == (8, 1)

        # Actions should respect bounds
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)
        assert torch.all(mean_action >= -1.0)
        assert torch.all(mean_action <= 1.0)

    def test_sample_gradient_flow(self, test_populations):
        """Test gradient flow through sampling (reparameterization trick)."""
        actor = MultiPopulationStochasticActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        state = torch.randn(4, 2, requires_grad=True)
        pop_states = {
            "pop1": torch.randn(4, 20),
            "pop2": torch.randn(4, 30),
        }

        action, _log_prob, _ = actor.sample(state, pop_states)
        loss = action.sum()
        loss.backward()

        assert state.grad is not None

    def test_deterministic_evaluation(self, test_populations):
        """Test that mean_action can be used for deterministic evaluation."""
        actor = MultiPopulationStochasticActor(
            pop_id="pop1",
            state_dim=2,
            action_dim=1,
            action_bounds=(-1.0, 1.0),
            population_configs=test_populations,
        )

        state = torch.randn(1, 2)
        pop_states = {
            "pop1": torch.randn(1, 20),
            "pop2": torch.randn(1, 30),
        }

        _, _, mean_action1 = actor.sample(state, pop_states)
        _, _, _mean_action2 = actor.sample(state, pop_states)

        # Mean should be deterministic (same input → same mean)
        mean, _ = actor(state, pop_states)
        mean_action_direct = torch.tanh(mean) * actor.action_scale + actor.action_bias

        assert torch.allclose(mean_action1, mean_action_direct, atol=1e-5)
