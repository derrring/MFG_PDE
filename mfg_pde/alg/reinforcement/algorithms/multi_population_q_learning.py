"""
Multi-Population Q-Learning for Heterogeneous Mean Field Games.

Extends Mean Field Q-Learning to K agent types with:
- Type-specific Q-networks Q^k(s, a, m^1, ..., m^K)
- Multi-population state m = (m^1, ..., m^K)
- Nash equilibrium learning through best-response dynamics

Mathematical Framework:
- Each type k has Q-function: Q^k(s^k, a^k, m)
- Best response: π^k = argmax_a Q^k(s^k, a, m)
- Nash equilibrium: All types simultaneously play best responses

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.alg.reinforcement.environments.multi_population_maze_env import (
        MultiPopulationMazeEnvironment,
    )

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


class MultiPopulationQNetwork(nn.Module):
    """
    Q-Network for single agent type in multi-population MFG.

    Architecture:
    - State encoder: Maps individual state to features
    - Population encoders: Separate encoder for each population type
    - Q-head: Combines features to output Q-values for actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dims: dict[str, int],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize multi-population Q-network.

        Args:
            state_dim: Dimension of individual state
            action_dim: Number of actions
            population_dims: Dict {type_id: pop_dim} for all K types
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dims = population_dims
        self.K = len(population_dims)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
        )

        # Separate encoder for each population type
        self.pop_encoders = nn.ModuleDict()
        for type_id, pop_dim in population_dims.items():
            self.pop_encoders[type_id] = nn.Sequential(
                nn.Linear(pop_dim, hidden_dims[0] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2),
                nn.ReLU(),
            )

        # Combined feature dimension
        combined_dim = hidden_dims[0] + self.K * (hidden_dims[0] // 2)

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )

    def forward(self, state: torch.Tensor, population_states: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Individual state [batch, state_dim]
            population_states: Dict {type_id: [batch, pop_dim]} for all K types

        Returns:
            Q-values [batch, action_dim]
        """
        # Encode individual state
        state_feat = self.state_encoder(state)

        # Encode all population states
        pop_feats = []
        for type_id in sorted(self.population_dims.keys()):
            pop_feat = self.pop_encoders[type_id](population_states[type_id])
            pop_feats.append(pop_feat)

        # Combine all features
        combined = torch.cat([state_feat, *pop_feats], dim=1)

        # Output Q-values
        return self.q_head(combined)


class ReplayBuffer:
    """Experience replay buffer for multi-population Q-learning."""

    def __init__(self, capacity: int, state_dim: int, num_population_types: int, pop_dim: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state
            num_population_types: Number of population types (K)
            pop_dim: Dimension of each population state
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.K = num_population_types
        self.pop_dim = pop_dim

        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Multi-population states (store as flat array)
        self.population_states = np.zeros((capacity, num_population_types, pop_dim), dtype=np.float32)
        self.next_population_states = np.zeros((capacity, num_population_types, pop_dim), dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        population_state: dict[str, NDArray],
        next_population_state: dict[str, NDArray],
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # Store population states (convert dict to array)
        for k, type_id in enumerate(sorted(population_state.keys())):
            self.population_states[idx, k] = population_state[type_id]
            self.next_population_states[idx, k] = next_population_state[type_id]

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "population_states": self.population_states[indices],
            "next_population_states": self.next_population_states[indices],
        }

    def __len__(self) -> int:
        return self.size


class MultiPopulationQLearning:
    """
    Multi-Population Q-Learning algorithm for heterogeneous MFG.

    Trains separate Q-networks for each agent type:
        Q^k(s^k, a^k, m^1, ..., m^K)

    Each type learns best response to current population distribution.
    """

    def __init__(
        self,
        env: MultiPopulationMazeEnvironment,
        type_id: str,
        state_dim: int,
        action_dim: int,
        population_dims: dict[str, int],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize Multi-Population Q-Learning for single agent type.

        Args:
            env: Multi-population maze environment
            type_id: Agent type this algorithm controls
            state_dim: State dimension for this type
            action_dim: Action dimension for this type
            population_dims: Dict {type_id: pop_dim} for ALL types
            config: Algorithm configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.env = env
        self.type_id = type_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dims = population_dims
        self.K = len(population_dims)

        # Default config
        default_config = {
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "batch_size": 64,
            "replay_buffer_size": 100000,
            "target_update_frequency": 100,
            "hidden_dims": [128, 128],
        }
        self.config = {**default_config, **(config or {})}

        # Q-networks
        self.q_network = MultiPopulationQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dims=population_dims,
            hidden_dims=self.config["hidden_dims"],
        )

        self.target_network = MultiPopulationQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dims=population_dims,
            hidden_dims=self.config["hidden_dims"],
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config["learning_rate"])

        # Replay buffer
        pop_dim = sum(population_dims.values())  # Total flattened population dim
        self.replay_buffer = ReplayBuffer(
            capacity=self.config["replay_buffer_size"],
            state_dim=state_dim,
            num_population_types=self.K,
            pop_dim=pop_dim // self.K,  # Average per type
        )

        # Training stats
        self.epsilon = self.config["epsilon"]
        self.update_count = 0

    def select_action(self, state: NDArray, population_states: dict[str, NDArray], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Individual state [state_dim]
            population_states: Dict {type_id: pop_state}
            training: If True, use epsilon-greedy; else greedy

        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        # Greedy action
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_states_t = {
                type_id: torch.FloatTensor(pop_state).unsqueeze(0) for type_id, pop_state in population_states.items()
            }
            q_values = self.q_network(state_t, pop_states_t)
            return q_values.argmax().item()

    def update(self) -> float | None:
        """Update Q-network from replay buffer."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.config["batch_size"])

        states = torch.FloatTensor(batch["states"])
        actions = torch.LongTensor(batch["actions"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        dones = torch.BoolTensor(batch["dones"])

        # Convert population states to dict format
        population_states = {}
        next_population_states = {}
        for k, type_id in enumerate(sorted(self.population_dims.keys())):
            population_states[type_id] = torch.FloatTensor(batch["population_states"][:, k, :])
            next_population_states[type_id] = torch.FloatTensor(batch["next_population_states"][:, k, :])

        # Compute current Q-values
        q_values = self.q_network(states, population_states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_population_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.config["discount_factor"] * max_next_q * (~dones))

        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.config["target_update_frequency"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.config["epsilon_decay"], self.config["epsilon_min"])

        return loss.item()

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train Q-network for this agent type.

        Note: This trains only ONE type. For Nash equilibrium, need to
        alternate training across all types.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0

            # Get initial state for this type
            state = observations[self.type_id][0]  # First agent of this type

            done = False
            while not done:
                # Get multi-population state
                multi_pop_state = self.env.get_multi_population_state()
                pop_states = {
                    type_id: density.flatten() for type_id, density in multi_pop_state.get_all_densities().items()
                }

                # Select action
                action = self.select_action(state, pop_states, training=True)

                # Execute action (for all agents of this type, for simplicity use same action)
                num_agents = len(observations[self.type_id])
                actions_dict = {
                    type_id: (
                        np.array([action] * num_agents)
                        if type_id == self.type_id
                        else np.random.randint(0, self.env.action_spaces[type_id].n, size=len(obs))
                    )
                    for type_id, obs in observations.items()
                }

                next_observations, rewards, terminated, truncated, _ = self.env.step(actions_dict)

                # Get reward for first agent of this type
                reward = rewards[self.type_id][0]
                next_state = next_observations[self.type_id][0]

                # Get next multi-population state
                next_multi_pop_state = self.env.get_multi_population_state()
                next_pop_states = {
                    type_id: density.flatten() for type_id, density in next_multi_pop_state.get_all_densities().items()
                }

                # Store transition
                self.replay_buffer.push(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    population_state=pop_states,
                    next_population_state=next_pop_states,
                    done=terminated or truncated,
                )

                # Update
                loss = self.update()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                observations = next_observations
                episode_reward += reward
                episode_length += 1

                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if (episode + 1) % 100 == 0:
                print(
                    f"[{self.type_id}] Episode {episode + 1}/{num_episodes}: "
                    f"Reward={np.mean(episode_rewards[-100:]):.2f}, "
                    f"Length={np.mean(episode_lengths[-100:]):.1f}, "
                    f"Epsilon={self.epsilon:.3f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
        }


def create_multi_population_q_learning_solvers(
    env: MultiPopulationMazeEnvironment,
    state_dims: dict[str, int],
    action_dims: dict[str, int],
    population_dims: dict[str, int],
    config: dict[str, Any] | None = None,
) -> dict[str, MultiPopulationQLearning]:
    """
    Create Q-learning solvers for all agent types.

    Args:
        env: Multi-population environment
        state_dims: Dict {type_id: state_dim}
        action_dims: Dict {type_id: action_dim}
        population_dims: Dict {type_id: pop_dim}
        config: Shared configuration

    Returns:
        Dict {type_id: solver}
    """
    solvers = {}

    for type_id in env.agent_types:
        solvers[type_id] = MultiPopulationQLearning(
            env=env,
            type_id=type_id,
            state_dim=state_dims[type_id],
            action_dim=action_dims[type_id],
            population_dims=population_dims,
            config=config,
        )

    return solvers
