"""
Multi-Population Deep Deterministic Policy Gradient for Mean Field Games.

Extends DDPG to multiple interacting populations with:
- Population-specific actors: μᵢ(s, m₁, ..., mₙ) → aᵢ ∈ ℝᵈⁱ
- Cross-population critics: Qᵢ(s, a, m₁, ..., mₙ)
- Independent replay buffers per population
- Heterogeneous action spaces supported
- Nash equilibrium convergence

Mathematical Framework:
- N populations with policies π₁, π₂, ..., πₙ
- Coupled Q-functions: Qᵢ(s, a, m₁, ..., mₙ) for population i
- Policy gradient for population i: ∇_θᵢ J = E[∇_θᵢ μᵢ(s,m) ∇_a Qᵢ(s,a,m)|_{a=μᵢ}]
- Nash equilibrium: Each population best-responds to others

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


class MultiPopulationDDPGActor(nn.Module):
    """
    Deterministic policy network for one population in multi-population setting.

    Architecture:
    - State encoder: s → features
    - Multi-population encoder: (m₁, m₂, ..., mₙ) → features
    - Action head: features → aᵢ ∈ ℝᵈⁱ (bounded by tanh)

    Key difference from single-population:
    - Takes all population states as input (not just own population)
    - Enables cross-population strategic interactions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dims: list[int],
        action_bounds: tuple[float, float],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize multi-population DDPG actor.

        Args:
            state_dim: Dimension of individual state
            action_dim: Dimension of continuous action for this population
            population_dims: Dimensions of all population states [dim₁, dim₂, ..., dimₙ]
            action_bounds: (min, max) action bounds for this population
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dims = population_dims
        self.total_pop_dim = sum(population_dims)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()
        )

        # Multi-population encoder (handles all populations)
        self.pop_encoder = nn.Sequential(
            nn.Linear(self.total_pop_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
        )

        # Action head
        combined_dim = hidden_dims[1] + hidden_dims[1] // 2
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]), nn.ReLU(), nn.Linear(hidden_dims[1], action_dim), nn.Tanh()
        )

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state: torch.Tensor, population_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Individual state [batch, state_dim]
            population_states: Concatenated population states [batch, total_pop_dim]

        Returns:
            Continuous action [batch, action_dim]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.pop_encoder(population_states)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        action = self.action_head(combined)
        return action * self.action_scale + self.action_bias


class MultiPopulationDDPGCritic(nn.Module):
    """
    Q-function network for one population with cross-population awareness.

    Architecture:
    - State encoder: s → features
    - Action encoder: a → features
    - Multi-population encoder: (m₁, m₂, ..., mₙ) → features
    - Q-head: combined → Qᵢ(s, a, m₁, ..., mₙ)

    Key difference from single-population:
    - Observes all population states for strategic Q-value estimation
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dims: list[int],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize multi-population DDPG critic.

        Args:
            state_dim: Dimension of individual state
            action_dim: Dimension of continuous action
            population_dims: Dimensions of all population states
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.total_pop_dim = sum(population_dims)

        # State encoder
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dims[0]), nn.ReLU())

        # Action encoder
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU())

        # Multi-population encoder
        self.pop_encoder = nn.Sequential(nn.Linear(self.total_pop_dim, hidden_dims[1]), nn.ReLU())

        # Q-value head
        combined_dim = hidden_dims[0] + 64 + hidden_dims[1]
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, population_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Individual state [batch, state_dim]
            action: Continuous action [batch, action_dim]
            population_states: Concatenated population states [batch, total_pop_dim]

        Returns:
            Q-value [batch]
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_states)
        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration.

    Generates temporally correlated noise for continuous action exploration.
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """
        Initialize OU noise process.

        Args:
            action_dim: Dimension of action space
            mu: Mean reversion level
            theta: Mean reversion speed
            sigma: Noise volatility
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        """Reset noise to initial state."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> NDArray:
        """Sample noise from OU process."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class MultiPopulationReplayBuffer:
    """Experience replay buffer for one population in multi-population setting."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, total_pop_dim: int):
        """
        Initialize replay buffer for one population.

        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Continuous action dimension
            total_pop_dim: Sum of all population state dimensions
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.population_states = np.zeros((capacity, total_pop_dim), dtype=np.float32)
        self.next_population_states = np.zeros((capacity, total_pop_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
        next_state: NDArray,
        population_states: NDArray,
        next_population_states: NDArray,
        done: bool,
    ):
        """Add transition to buffer."""
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.population_states[idx] = population_states
        self.next_population_states[idx] = next_population_states
        self.dones[idx] = done

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
            "population_states": self.population_states[indices],
            "next_population_states": self.next_population_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class MultiPopulationDDPG:
    """
    Multi-Population Deep Deterministic Policy Gradient for MFG.

    Extends DDPG to handle N interacting populations with:
    - Population-specific deterministic actors
    - Population-specific critics with cross-population awareness
    - Independent replay buffers and exploration noise
    - Heterogeneous action spaces
    - Nash equilibrium convergence

    Usage:
        env = MultiPopulationMFGEnv(num_populations=3, ...)
        algo = MultiPopulationDDPG(
            env=env,
            num_populations=3,
            state_dims=[2, 2, 2],
            action_dims=[2, 2, 1],
            population_dims=[100, 100, 100],
            action_bounds=[(-1, 1), (-2, 2), (0, 1)]
        )
        stats = algo.train(num_episodes=1000)
    """

    def __init__(
        self,
        env: Any,
        num_populations: int,
        state_dims: list[int] | int,
        action_dims: list[int],
        population_dims: list[int] | int,
        action_bounds: list[tuple[float, float]],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize Multi-Population DDPG.

        Args:
            env: Multi-population MFG environment
            num_populations: Number of interacting populations (N ≥ 2)
            state_dims: State dimension per population (list) or shared (int)
            action_dims: Action dimension per population [d₁, d₂, ..., dₙ]
            population_dims: Population state dimension per population (list) or shared (int)
            action_bounds: Action bounds per population [(min₁, max₁), ...]
            config: Algorithm configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        if num_populations < 2:
            raise ValueError(f"Multi-population requires N ≥ 2, got {num_populations}")

        self.env = env
        self.num_populations = num_populations

        # Handle state dimensions
        if isinstance(state_dims, int):
            self.state_dims = [state_dims] * num_populations
        else:
            if len(state_dims) != num_populations:
                raise ValueError(f"state_dims length mismatch: {len(state_dims)} != {num_populations}")
            self.state_dims = state_dims

        # Handle action dimensions
        if len(action_dims) != num_populations:
            raise ValueError(f"action_dims length mismatch: {len(action_dims)} != {num_populations}")
        self.action_dims = action_dims

        # Handle population dimensions
        if isinstance(population_dims, int):
            self.population_dims = [population_dims] * num_populations
        else:
            if len(population_dims) != num_populations:
                raise ValueError(f"population_dims length mismatch: {len(population_dims)} != {num_populations}")
            self.population_dims = population_dims

        # Handle action bounds
        if len(action_bounds) != num_populations:
            raise ValueError(f"action_bounds length mismatch: {len(action_bounds)} != {num_populations}")
        self.action_bounds = action_bounds

        # Default config
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,  # Soft update
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "ou_theta": 0.15,
            "ou_sigma": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        # Initialize networks for each population
        self.actors: list[MultiPopulationDDPGActor] = []
        self.actor_targets: list[MultiPopulationDDPGActor] = []
        self.critics: list[MultiPopulationDDPGCritic] = []
        self.critic_targets: list[MultiPopulationDDPGCritic] = []
        self.actor_optimizers: list[Any] = []
        self.critic_optimizers: list[Any] = []
        self.replay_buffers: list[MultiPopulationReplayBuffer] = []
        self.noises: list[OrnsteinUhlenbeckNoise] = []

        total_pop_dim = sum(self.population_dims)

        for pop_id in range(num_populations):
            # Actor
            actor = MultiPopulationDDPGActor(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                action_bounds=self.action_bounds[pop_id],
                hidden_dims=self.config["hidden_dims"],
            )
            actor_target = MultiPopulationDDPGActor(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                action_bounds=self.action_bounds[pop_id],
                hidden_dims=self.config["hidden_dims"],
            )
            actor_target.load_state_dict(actor.state_dict())

            # Critic
            critic = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )
            critic_target = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )
            critic_target.load_state_dict(critic.state_dict())

            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.config["actor_lr"])
            critic_optimizer = optim.Adam(critic.parameters(), lr=self.config["critic_lr"])

            # Replay buffer
            replay_buffer = MultiPopulationReplayBuffer(
                capacity=self.config["replay_buffer_size"],
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                total_pop_dim=total_pop_dim,
            )

            # Exploration noise
            noise = OrnsteinUhlenbeckNoise(
                action_dim=self.action_dims[pop_id], theta=self.config["ou_theta"], sigma=self.config["ou_sigma"]
            )

            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
            self.replay_buffers.append(replay_buffer)
            self.noises.append(noise)

        # Training stats
        self.update_count = 0

    def select_actions(
        self, states: dict[int, NDArray], population_states: dict[int, NDArray], training: bool = True
    ) -> dict[int, NDArray]:
        """
        Select actions for all populations.

        Args:
            states: {pop_id: state} individual states
            population_states: {pop_id: pop_state} population distributions
            training: If True, add exploration noise

        Returns:
            {pop_id: action} actions for each population
        """
        # Concatenate all population states
        pop_states_concat = np.concatenate([population_states[i] for i in range(self.num_populations)])

        actions = {}
        for pop_id in range(self.num_populations):
            state = states[pop_id]
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                pop_state_t = torch.FloatTensor(pop_states_concat).unsqueeze(0)
                action = self.actors[pop_id](state_t, pop_state_t).squeeze(0).numpy()

            if training:
                noise = self.noises[pop_id].sample()
                bounds = self.action_bounds[pop_id]
                action = np.clip(action + noise, bounds[0], bounds[1])

            actions[pop_id] = action

        return actions

    def update(self) -> dict[int, tuple[float, float]] | None:
        """
        Update all population actors and critics.

        Returns:
            {pop_id: (critic_loss, actor_loss)} or None if buffers too small
        """
        # Check if all buffers have enough data
        for buffer in self.replay_buffers:
            if len(buffer) < self.config["batch_size"]:
                return None

        losses = {}

        for pop_id in range(self.num_populations):
            # Sample batch for this population
            batch = self.replay_buffers[pop_id].sample(self.config["batch_size"])

            states = torch.FloatTensor(batch["states"])
            actions = torch.FloatTensor(batch["actions"])
            rewards = torch.FloatTensor(batch["rewards"])
            next_states = torch.FloatTensor(batch["next_states"])
            population_states = torch.FloatTensor(batch["population_states"])
            next_population_states = torch.FloatTensor(batch["next_population_states"])
            dones = torch.BoolTensor(batch["dones"])

            # Update critic
            with torch.no_grad():
                next_actions = self.actor_targets[pop_id](next_states, next_population_states)
                target_q = self.critic_targets[pop_id](next_states, next_actions, next_population_states)
                target_q = rewards + self.config["discount_factor"] * target_q * (~dones)

            current_q = self.critics[pop_id](states, actions, population_states)
            critic_loss = nn.functional.mse_loss(current_q, target_q)

            self.critic_optimizers[pop_id].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[pop_id].step()

            # Update actor
            actor_actions = self.actors[pop_id](states, population_states)
            actor_loss = -self.critics[pop_id](states, actor_actions, population_states).mean()

            self.actor_optimizers[pop_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[pop_id].step()

            # Soft update target networks
            self._soft_update(self.actors[pop_id], self.actor_targets[pop_id])
            self._soft_update(self.critics[pop_id], self.critic_targets[pop_id])

            losses[pop_id] = (critic_loss.item(), actor_loss.item())

        self.update_count += 1
        return losses

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train multi-population DDPG agents.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training statistics per population
        """
        # Per-population tracking
        episode_rewards: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        critic_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        actor_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}

        episode_lengths = []

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward_dict = dict.fromkeys(range(self.num_populations), 0.0)
            episode_length = 0

            # Reset noise for all populations
            for noise in self.noises:
                noise.reset()

            done = False
            while not done:
                # Get population states
                population_states = self.env.get_population_states()

                # Select actions for all populations
                actions = self.select_actions(states, population_states, training=True)

                # Execute actions
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                next_population_states = self.env.get_population_states()

                # Concatenate population states for storage
                pop_states_concat = np.concatenate([population_states[i] for i in range(self.num_populations)])
                next_pop_states_concat = np.concatenate(
                    [next_population_states[i] for i in range(self.num_populations)]
                )

                # Store transitions for each population
                for pop_id in range(self.num_populations):
                    reward_val = float(rewards[pop_id])
                    self.replay_buffers[pop_id].push(
                        state=states[pop_id],
                        action=actions[pop_id],
                        reward=reward_val,
                        next_state=next_states[pop_id],
                        population_states=pop_states_concat,
                        next_population_states=next_pop_states_concat,
                        done=terminated[pop_id] or truncated[pop_id],
                    )
                    episode_reward_dict[pop_id] += reward_val

                # Update all populations
                losses = self.update()
                if losses is not None:
                    for pop_id, (critic_loss, actor_loss) in losses.items():
                        critic_losses[pop_id].append(critic_loss)
                        actor_losses[pop_id].append(actor_loss)

                states = next_states
                episode_length += 1

                # Check if any population is done
                done = any(terminated.values()) or any(truncated.values())

            # Record episode rewards
            for pop_id in range(self.num_populations):
                episode_rewards[pop_id].append(episode_reward_dict[pop_id])
            episode_lengths.append(episode_length)

            # Logging
            if (episode + 1) % 100 == 0:
                print(f"\nEpisode {episode + 1}/{num_episodes}:")
                print(f"  Length: {np.mean(episode_lengths[-100:]):.1f}")
                for pop_id in range(self.num_populations):
                    avg_reward = np.mean(episode_rewards[pop_id][-100:])
                    avg_critic = np.mean(critic_losses[pop_id][-100:]) if critic_losses[pop_id] else 0
                    avg_actor = np.mean(actor_losses[pop_id][-100:]) if actor_losses[pop_id] else 0
                    print(
                        f"  Pop {pop_id}: Reward={avg_reward:.2f}, "
                        f"Critic Loss={avg_critic:.4f}, Actor Loss={avg_actor:.4f}"
                    )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
        }
