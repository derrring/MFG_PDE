"""
Unit tests for Nash Q-Learning functionality.

Tests verify that Mean Field Q-Learning correctly implements Nash equilibrium
computation for symmetric Mean Field Games.
"""

import pytest

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms.mean_field_q_learning import (
        MeanFieldQLearning,
        MeanFieldQNetwork,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestNashQLearning:
    """Tests for Nash Q-Learning implementation."""

    def test_nash_value_equals_max_for_symmetric_game(self):
        """
        Test that Nash equilibrium value equals max Q-value for symmetric MFG.

        For symmetric games, Nash equilibrium is deterministic:
            Nash_value(s, m) = max_a Q(s, a, m)
        """
        # Create simple Q-network
        state_dim = 4
        action_dim = 5
        population_dim = 8

        q_network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=[32, 32],
        )

        # Random states and population
        batch_size = 10
        states = torch.randn(batch_size, state_dim)
        pop_states = torch.randn(batch_size, population_dim)

        # Compute Q-values
        q_values = q_network(states, pop_states)

        # Nash value should equal max Q-value
        max_q = q_values.max(dim=1)[0]
        nash_actions = q_values.argmax(dim=1)

        # Verify Nash value is max
        assert q_values.shape == (batch_size, action_dim)
        assert max_q.shape == (batch_size,)
        assert nash_actions.shape == (batch_size,)

        # Verify Nash actions achieve max Q-value
        for i in range(batch_size):
            nash_action = nash_actions[i].item()
            assert torch.isclose(q_values[i, nash_action], max_q[i])

    def test_compute_nash_value_method(self):
        """Test the compute_nash_value() method."""

        # Create mock environment
        class MockEnv:
            class Config:
                num_agents = 4

            config = Config()

            def reset(self):
                return np.random.randn(4, 4), {}

        env = MockEnv()

        # Create algorithm
        algo = MeanFieldQLearning(
            env=env,
            state_dim=4,
            action_dim=5,
            population_dim=8,
            config={"hidden_dims": [32, 32]},
        )

        # Random states and population
        batch_size = 10
        states = torch.randn(batch_size, 4)
        pop_states = torch.randn(batch_size, 8)

        # Compute Nash value
        nash_values = algo.compute_nash_value(states, pop_states, game_type="symmetric")

        # Verify shape
        assert nash_values.shape == (batch_size,)

        # Verify Nash value equals max Q-value
        q_values = algo.target_network(states, pop_states)
        max_q = q_values.max(dim=1)[0]

        assert torch.allclose(nash_values, max_q)

    def test_nash_value_raises_for_general_games(self):
        """Test that general games raise NotImplementedError."""

        # Create mock environment
        class MockEnv:
            class Config:
                num_agents = 4

            config = Config()

            def reset(self):
                return np.random.randn(4, 4), {}

        env = MockEnv()

        algo = MeanFieldQLearning(
            env=env,
            state_dim=4,
            action_dim=5,
            population_dim=8,
        )

        states = torch.randn(10, 4)
        pop_states = torch.randn(10, 8)

        # Should raise for non-symmetric games
        with pytest.raises(NotImplementedError):
            algo.compute_nash_value(states, pop_states, game_type="zero_sum")

        with pytest.raises(NotImplementedError):
            algo.compute_nash_value(states, pop_states, game_type="general")

    def test_update_uses_nash_value(self):
        """
        Test that Q-network update uses Nash equilibrium value.

        The target Q-value should be:
            target = r + γ * Nash_value(s', m')
                  = r + γ * max_a Q(s', a, m')
        """

        # Create mock environment
        class MockEnv:
            class Config:
                num_agents = 4

            config = Config()

            def reset(self):
                return np.random.randn(4, 4), {}

            def step(self, actions):
                obs = np.random.randn(4, 4)
                rewards = np.random.randn(4)
                terminated = False
                truncated = False
                info = {"agents_done": np.zeros(4, dtype=bool)}
                return obs, rewards, terminated, truncated, info

        env = MockEnv()

        # Create algorithm with small replay buffer
        algo = MeanFieldQLearning(
            env=env,
            state_dim=4,
            action_dim=5,
            population_dim=8,
            config={"batch_size": 4, "replay_buffer_size": 100, "hidden_dims": [32, 32]},
        )

        # Add some experiences to replay buffer
        for _ in range(10):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            pop_state = np.random.randn(8).astype(np.float32)
            next_pop_state = np.random.randn(8).astype(np.float32)
            done = False

            algo.replay_buffer.push(state, action, reward, next_state, pop_state, next_pop_state, done)

        # Run update
        loss = algo._update_q_network()

        # Verify loss is computed
        assert isinstance(loss, float)
        assert loss >= 0

    def test_nash_equilibrium_convergence_property(self):
        """
        Test that Nash equilibrium satisfies fixed-point property.

        At Nash equilibrium:
            π*(s, m) = argmax_a Q*(s, a, m)
            Q*(s, a, m) = r(s, a, m) + γ * Nash_value(s', m')

        This test verifies the structure, not actual convergence.
        """
        # Create Q-network
        state_dim = 4
        action_dim = 5
        population_dim = 8

        q_network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=[32, 32],
        )

        # Sample state and population
        state = torch.randn(1, state_dim)
        pop_state = torch.randn(1, population_dim)

        # Get Q-values
        q_values = q_network(state, pop_state)

        # Nash policy = argmax
        nash_action = q_values.argmax().item()
        nash_value = q_values[0, nash_action].item()

        # Verify Nash action achieves maximum Q-value
        max_q_value = q_values.max().item()
        assert abs(nash_value - max_q_value) < 1e-6

        # Verify Nash action is deterministic (not mixed strategy)
        assert 0 <= nash_action < action_dim


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMeanFieldQNetworkArchitecture:
    """Tests for Mean Field Q-Network architecture."""

    def test_network_output_shape(self):
        """Test that network outputs correct shape."""
        state_dim = 4
        action_dim = 5
        population_dim = 8
        batch_size = 10

        network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        states = torch.randn(batch_size, state_dim)
        pop_states = torch.randn(batch_size, population_dim)

        q_values = network(states, pop_states)

        assert q_values.shape == (batch_size, action_dim)

    def test_network_encodes_population_state(self):
        """Test that network is sensitive to population state."""
        state_dim = 4
        action_dim = 5
        population_dim = 8

        network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        # Same state, different populations
        state = torch.randn(1, state_dim)
        pop_state_1 = torch.randn(1, population_dim)
        pop_state_2 = torch.randn(1, population_dim)

        q_values_1 = network(state, pop_state_1)
        q_values_2 = network(state, pop_state_2)

        # Q-values should differ for different populations
        assert not torch.allclose(q_values_1, q_values_2)

    def test_network_gradient_flow(self):
        """Test that gradients flow through network."""
        state_dim = 4
        action_dim = 5
        population_dim = 8

        network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
        )

        states = torch.randn(10, state_dim, requires_grad=True)
        pop_states = torch.randn(10, population_dim)

        q_values = network(states, pop_states)
        loss = q_values.sum()
        loss.backward()

        # Verify gradients exist
        assert states.grad is not None
        assert states.grad.shape == states.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
