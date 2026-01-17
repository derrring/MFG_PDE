"""
Mean Field Q-Learning Algorithm.

This module implements a Q-learning approach for Mean Field Games, where agents
learn Q-functions that depend on both individual state-action pairs and the
population state (mean field).

Key Features:
- State-action value function: Q(s, a, m)
- Population state tracking and updates
- Experience replay for stability
- Nash equilibrium detection

Mathematical Framework:
- Q-function: Q(s, a, m) = E[∑ γ^t r(s_t, a_t, m_t) | s_0=s, a_0=a, m_0=m]
- Population consistency: m = μ(π)
- Nash condition: π(s, m) ∈ argmax_a Q(s, a, m)

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_f
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mfg_pde.utils.mfg_logging.logger import get_logger

logger = get_logger(__name__)


class MeanFieldQNetwork(nn.Module):
    """
    Neural network for Mean Field Q-function.

    Architecture:
    - State encoder: encodes individual agent state
    - Population encoder: encodes population state (mean field)
    - Fusion layer: combines individual and population features
    - Q-value head: outputs Q-values for each action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            self.activation,
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            self.activation,
        )

        # Fusion and Q-value layers
        fusion_input_dim = hidden_dims[0]  # state_features + population_features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[1]),
            self.activation,
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            self.activation,
        )

        # Q-value head
        self.q_head = nn.Linear(hidden_dims[1], action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, population_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Individual agent state [batch_size, state_dim]
            population_state: Population state [batch_size, population_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        # Encode individual state
        state_features = self.state_encoder(state)

        # Encode population state
        population_features = self.population_encoder(population_state)

        # Fuse features
        combined_features = torch.cat([state_features, population_features], dim=1)
        fused_features = self.fusion_layers(combined_features)

        # Compute Q-values
        q_values = self.q_head(fused_features)

        return q_values


class ExperienceReplay:
    """Experience replay buffer for MFRL."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        population_state: np.ndarray,
        next_population_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, population_state, next_population_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        experiences = random.sample(self.buffer, batch_size)

        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        population_states = np.array([e[4] for e in experiences])
        next_population_states = np.array([e[5] for e in experiences])
        dones = np.array([e[6] for e in experiences])

        return states, actions, rewards, next_states, population_states, next_population_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class MeanFieldQLearning:
    """
    Mean Field Q-Learning algorithm for MFG problems.

    This algorithm learns Q-functions that depend on both individual agent
    states and the population state (mean field), enabling coordination
    in large multi-agent systems.

    Mathematical Framework:
        Q-function: Q(s, a, m) = E[∑ γ^t r(s_t, a_t, m_t) | s_0=s, a_0=a, m_0=m]
        Population consistency: m = μ(π)
        Nash equilibrium: π(s, m) ∈ argmax_a Q(s, a, m)

    Nash Q-Learning Interpretation:
        For symmetric Mean Field Games, this algorithm is equivalent to Nash Q-Learning,
        since the Nash equilibrium reduces to the best response to the mean field.
        The max operation in the target value computation (see _update_q_network())
        implements the Nash equilibrium value for symmetric games:

            Nash_value(s', m') = max_a Q(s', a, m')

        For heterogeneous or competitive multi-agent settings, a general Nash solver
        would be needed. See nash_q_learning_formulation.md for details.
    """

    def __init__(
        self,
        env: Any,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Mean Field Q-Learning algorithm."""
        if not TORCH_AVAILABLE:
            raise ImportError("Mean Field Q-Learning requires PyTorch. Install with: pip install torch")

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim
        self.logger = get_logger(__name__)

        # Configuration
        default_config = {
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "batch_size": 64,
            "target_update_frequency": 100,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 256],
            "activation": "relu",
        }
        self.config = {**default_config, **(config or {})}

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize networks
        self.q_network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
            activation=self.config["activation"],
        ).to(self.device)

        self.target_network = MeanFieldQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
            activation=self.config["activation"],
        ).to(self.device)

        # Copy main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config["learning_rate"])

        # Experience replay
        self.replay_buffer = ExperienceReplay(self.config["replay_buffer_size"])

        # Training state
        self.epsilon = self.config["epsilon"]
        self.training_step = 0

        # Results tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.loss_history: list[float] = []
        self.nash_errors: list[float] = []

    def select_action(self, state: np.ndarray, population_state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Individual agent state
            population_state: Population state (mean field)
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_dim - 1)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            population_tensor = torch.FloatTensor(population_state).unsqueeze(0).to(self.device)

            q_values = self.q_network(state_tensor, population_tensor)
            action = q_values.argmax().item()

        return action

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for all agents (evaluation mode).

        Args:
            observations: Observations for all agents

        Returns:
            Actions for all agents
        """
        obs_batch = self._ensure_batch(observations)
        num_agents = obs_batch.shape[0]
        actions = np.zeros(num_agents, dtype=int)

        # Compute population state
        population_state = self._compute_population_state(obs_batch)

        for i in range(num_agents):
            action = self.select_action(obs_batch[i], population_state, training=False)
            actions[i] = action

        return actions

    def compute_nash_value(
        self, state: torch.Tensor, population_state: torch.Tensor, game_type: str = "symmetric"
    ) -> torch.Tensor:
        """
        Compute Nash equilibrium value at given state and population.

        For symmetric MFG, the Nash equilibrium value is simply the maximum
        Q-value over all actions, since all agents follow the same best-response
        policy to the mean field.

        Args:
            state: Individual agent state [batch_size, state_dim]
            population_state: Population state [batch_size, population_dim]
            game_type: Type of game ("symmetric", "zero_sum", "general")

        Returns:
            Nash equilibrium values [batch_size]

        Note:
            Currently only symmetric games are supported. For general games,
            a Nash solver would be needed to compute mixed-strategy equilibria.
            See nash_q_learning_architecture.md for extension designs.
        """
        with torch.no_grad():
            q_values = self.target_network(state, population_state)

            if game_type == "symmetric":
                # Symmetric MFG: Nash value = max_a Q(s, a, m)
                nash_values = q_values.max(dim=1)[0]
            else:
                raise NotImplementedError(
                    f"Nash equilibrium computation for '{game_type}' games not yet implemented. "
                    "Currently only 'symmetric' games are supported. "
                    "For general games, see nash_q_learning_architecture.md for extension designs."
                )

        return nash_values

    def train(self, num_episodes: int) -> dict[str, Any]:
        """
        Train the Mean Field Q-Learning algorithm.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training results
        """
        self.logger.info(f"Starting Mean Field Q-Learning training for {num_episodes} episodes")

        for episode in range(num_episodes):
            _episode_reward, _episode_length = self._run_episode(episode)

            # Decay epsilon
            if self.epsilon > self.config["epsilon_min"]:
                self.epsilon *= self.config["epsilon_decay"]

            # Periodic logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
                self.logger.info(
                    f"Episode {episode}: Avg Reward = {avg_reward:.3f}, "
                    f"Avg Loss = {avg_loss:.6f}, Epsilon = {self.epsilon:.3f}"
                )

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "loss_history": self.loss_history,
            "nash_errors": self.nash_errors,
        }

    def _run_episode(self, episode_num: int) -> tuple[float, int]:
        """Run a single training episode."""
        obs, info = self.env.reset()
        obs_batch = self._ensure_batch(obs)
        total_reward = 0
        step_count = 0

        while True:
            # Compute population state
            population_state = self._compute_population_state(obs_batch)

            # Select actions for all agents
            num_agents = obs_batch.shape[0]
            actions = np.zeros(num_agents, dtype=int)
            for i in range(num_agents):
                actions[i] = self.select_action(obs_batch[i], population_state)

            # Execute actions
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            next_batch = self._ensure_batch(next_obs)
            next_population_state = self._compute_population_state(next_batch)

            # Store experiences for all agents
            agent_done_flags = info.get(
                "agents_done",
                np.full(num_agents, bool(terminated or truncated)),
            )

            for i in range(num_agents):
                self.replay_buffer.push(
                    state=obs_batch[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_batch[i],
                    population_state=population_state,
                    next_population_state=next_population_state,
                    done=bool(agent_done_flags[i]),
                )

            # Update Q-network
            if len(self.replay_buffer) >= self.config["batch_size"]:
                loss = self._update_q_network()
                self.loss_history.append(loss)

            # Update target network
            if self.training_step % self.config["target_update_frequency"] == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            total_reward += float(np.mean(rewards))
            step_count += 1
            obs_batch = next_batch
            self.training_step += 1

            if terminated or truncated:
                break

        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step_count)
        self.nash_errors.append(info.get("nash_error", 0.0))

        return total_reward, step_count

    def _update_q_network(self) -> float:
        """
        Update Q-network using experience replay.

        This implements the Nash Q-Learning update for symmetric Mean Field Games.
        The max operation below computes the Nash equilibrium value for symmetric games.
        """
        # Sample batch
        states, actions, rewards, next_states, pop_states, next_pop_states, dones = self.replay_buffer.sample(
            self.config["batch_size"]
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        pop_states = torch.FloatTensor(pop_states).to(self.device)
        next_pop_states = torch.FloatTensor(next_pop_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values: Q(s, a, m)
        current_q_values = self.q_network(states, pop_states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Nash equilibrium value for symmetric MFG (using target network for stability)
        with torch.no_grad():
            # Nash value = max_a Q(s', a, m') for symmetric games
            next_q_values = self.target_network(next_states, next_pop_states).max(1)[0]
            target_q_values = rewards + (self.config["discount_factor"] * next_q_values * (~dones))

        # Compute loss and optimize
        loss = torch_f.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_population_state(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute population state representation from observations.

        Args:
            observations: Observations for all agents

        Returns:
            Population state vector
        """
        obs_batch = self._ensure_batch(observations)
        mean_obs = np.mean(obs_batch, axis=0)
        std_obs = np.std(obs_batch, axis=0)
        population_state = np.concatenate([mean_obs, std_obs])

        # Ensure consistent dimensionality
        if len(population_state) != self.population_dim:
            # Pad or truncate to match expected dimension
            if len(population_state) < self.population_dim:
                padding = np.zeros(self.population_dim - len(population_state))
                population_state = np.concatenate([population_state, padding])
            else:
                population_state = population_state[: self.population_dim]

        return population_state

    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
                "epsilon": self.epsilon,
            },
            filepath,
        )
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        self.epsilon = checkpoint["epsilon"]

        self.logger.info(f"Model loaded from {filepath}")

    def _ensure_batch(self, observations: np.ndarray) -> np.ndarray:
        """Ensure observations are returned as batch-major arrays."""
        obs_batch = np.atleast_2d(observations).astype(np.float32)
        if obs_batch.shape[0] != self.env.config.num_agents:
            # When some agents have terminated we may receive fewer obs; pad with last observation
            last_obs = obs_batch[-1]
            pad_count = self.env.config.num_agents - obs_batch.shape[0]
            if pad_count > 0:
                padding = np.repeat(last_obs[None, :], pad_count, axis=0)
                obs_batch = np.vstack([obs_batch, padding])
        return obs_batch


def create_mean_field_q_learning(env, config: dict[str, Any] | None = None) -> MeanFieldQLearning:  # type: ignore[no-untyped-def]
    """
    Factory function to create Mean Field Q-Learning algorithm.

    Args:
        env: MFG environment
        config: Algorithm configuration

    Returns:
        Configured Mean Field Q-Learning instance
    """
    # Determine dimensions from environment
    obs, _ = env.reset()
    obs_batch = np.atleast_2d(obs).astype(np.float32)

    scenario = getattr(env.config, "scenario", None)

    if scenario == "crowd_navigation":
        state_dim = 6  # pos, vel, target
        action_dim = 4  # 4 discrete movement directions
        population_dim = 12  # mean + std of observations

    elif scenario == "linear_quadratic":
        state_dim = 4  # pos, mean_pos
        action_dim = 5  # 5 discrete control levels
        population_dim = 8  # mean + std of observations

    elif scenario == "finite_state":
        state_dim = 2  # state, state_density
        action_dim = 5  # stay, up, down, left, right
        population_dim = 4  # mean + std of observations

    elif scenario == "epidemic":
        state_dim = 4  # health_state, local_infection_rate, S_ratio, I_ratio
        action_dim = 2  # normal, isolate
        population_dim = 8  # mean + std of observations

    elif scenario == "price_formation":
        state_dim = 3  # holdings, cash, price
        action_dim = 11  # discrete buy/sell actions (-5 to +5)
        population_dim = 6  # mean + std of observations

    else:
        # Default dimensions from environment
        state_dim = obs_batch.shape[1]

        # Extract action_dim from environment's action space
        # Backend compatibility - gym environment API (Issue #543 acceptable)
        # hasattr checks for gym/gymnasium action space attributes
        if hasattr(env, "action_space"):  # Issue #543 acceptable
            if hasattr(env.action_space, "n"):  # Issue #543 acceptable
                # Discrete action space
                action_dim = env.action_space.n
            elif hasattr(env.action_space, "nvec"):  # Issue #543 acceptable
                # MultiDiscrete action space - use single agent's action dim
                action_dim = env.action_space.nvec[0]
            else:
                action_dim = 5  # Fallback
        else:
            action_dim = 5  # Fallback

        population_dim = state_dim * 2  # mean + std

    # Reset environment to initial state for downstream training
    env.reset()

    return MeanFieldQLearning(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        population_dim=population_dim,
        config=config,
    )
