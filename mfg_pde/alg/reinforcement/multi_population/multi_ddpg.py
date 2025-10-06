"""
Multi-Population DDPG for Mean Field Games.

Extends single-population DDPG to heterogeneous multi-population systems
where each population can have different state/action dimensions and
all populations interact through coupled mean field distributions.

Mathematical Framework:
- Population i: μ_i(s_i, m_1, ..., m_N) → a_i ∈ A_i
- Q-function: Q_i(s_i, a_i, m_1, ..., m_N)
- Nash equilibrium: All populations simultaneously optimize

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .base_environment import MultiPopulationMFGEnvironment
    from .population_config import PopulationConfig

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

from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import (
    OrnsteinUhlenbeckNoise,
    ReplayBuffer,
)

from .networks import MultiPopulationActor, MultiPopulationCritic


class MultiPopulationDDPG:
    """
    DDPG agent for multi-population Mean Field Games.

    Manages a single population in a multi-population system:
    - Observes all population distributions (m_1, ..., m_N)
    - Takes actions based on own state and joint mean field
    - Learns Q_i(s_i, a_i, m_1, ..., m_N) and μ_i(s_i, m_1, ..., m_N)
    """

    def __init__(
        self,
        pop_id: str,
        env: MultiPopulationMFGEnvironment,
        population_configs: dict[str, PopulationConfig],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize multi-population DDPG agent.

        Args:
            pop_id: ID of this population
            env: Multi-population MFG environment
            population_configs: All population configurations
            config: Algorithm configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.pop_id = pop_id
        self.env = env
        self.population_configs = population_configs

        # Get own configuration
        self.pop_config = population_configs[pop_id]
        self.state_dim = self.pop_config.state_dim
        self.action_dim = self.pop_config.action_dim
        self.action_bounds = self.pop_config.action_bounds

        # Default config
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "noise_theta": 0.15,
            "noise_sigma": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        # Compute population distribution dimension
        # Assume 10-bin histogram per state dimension
        self.pop_dist_dim = self.state_dim * 10

        # Actor networks
        self.actor = MultiPopulationActor(
            pop_id=pop_id,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_bounds=self.action_bounds,
            population_configs=population_configs,
            hidden_dims=self.config["hidden_dims"],
        )

        self.actor_target = MultiPopulationActor(
            pop_id=pop_id,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_bounds=self.action_bounds,
            population_configs=population_configs,
            hidden_dims=self.config["hidden_dims"],
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = MultiPopulationCritic(
            pop_id=pop_id,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            population_configs=population_configs,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic_target = MultiPopulationCritic(
            pop_id=pop_id,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            population_configs=population_configs,
            hidden_dims=self.config["hidden_dims"],
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            mu=np.zeros(self.action_dim),
            sigma=self.config["noise_sigma"],
            theta=self.config["noise_theta"],
        )

        # Replay buffer (stores joint population states)
        self.replay_buffer = ReplayBuffer(
            capacity=self.config["replay_buffer_size"],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            pop_dim=self._compute_joint_pop_dim(),
        )

    def _compute_joint_pop_dim(self) -> int:
        """Compute dimension of joint population state."""
        total_dim = 0
        for config in self.population_configs.values():
            total_dim += config.state_dim * 10  # 10-bin histogram
        return total_dim

    def select_action(
        self,
        state: NDArray,
        population_states: dict[str, NDArray],
        training: bool = True,
    ) -> NDArray:
        """
        Select action using actor with optional exploration noise.

        Args:
            state: Own state [state_dim]
            population_states: {pop_id: distribution [dist_dim]}
            training: If True, add OU noise

        Returns:
            Action [action_dim]
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Convert population states to tensors
            pop_states_t = {pop_id: torch.FloatTensor(dist).unsqueeze(0) for pop_id, dist in population_states.items()}

            action = self.actor(state_t, pop_states_t).squeeze(0).numpy()

        if training:
            noise_sample = self.noise.sample()
            action = np.clip(
                action + noise_sample,
                self.action_bounds[0],
                self.action_bounds[1],
            )

        return action

    def update(self) -> tuple[float, float] | None:
        """
        Update actor and critic networks.

        Returns:
            (critic_loss, actor_loss) or None if buffer too small
        """
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.config["batch_size"])

        states = torch.FloatTensor(batch["states"])
        actions = torch.FloatTensor(batch["actions"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        population_states = torch.FloatTensor(batch["population_states"])
        next_population_states = torch.FloatTensor(batch["next_population_states"])
        dones = torch.BoolTensor(batch["dones"])

        # Unpack joint population states
        pop_states_dict = self._unpack_population_states(population_states)
        next_pop_states_dict = self._unpack_population_states(next_population_states)

        # Compute TD target
        with torch.no_grad():
            next_actions = self.actor_target(next_states, next_pop_states_dict)
            target_q = self.critic_target(next_states, next_actions, next_pop_states_dict)
            target = rewards + self.config["discount_factor"] * target_q * (~dones)

        # Update critic
        current_q = self.critic(states, actions, pop_states_dict)
        critic_loss = nn.functional.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states, pop_states_dict)
        actor_loss = -self.critic(states, actor_actions, pop_states_dict).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _unpack_population_states(self, joint_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Unpack joint population state tensor to dictionary.

        Args:
            joint_states: Concatenated states [batch, joint_dim]

        Returns:
            {pop_id: distribution [batch, dist_dim]}
        """
        pop_states = {}
        offset = 0

        for pop_id in sorted(self.population_configs.keys()):
            dist_dim = self.population_configs[pop_id].state_dim * 10
            pop_states[pop_id] = joint_states[:, offset : offset + dist_dim]
            offset += dist_dim

        return pop_states

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def store_transition(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
        next_state: NDArray,
        population_states: dict[str, NDArray],
        next_population_states: dict[str, NDArray],
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            state: Own state
            action: Own action
            reward: Own reward
            next_state: Own next state
            population_states: All population distributions
            next_population_states: Next population distributions
            done: Terminal flag
        """
        # Pack population states into single array
        joint_pop_state = self._pack_population_states(population_states)
        joint_next_pop_state = self._pack_population_states(next_population_states)

        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            population_state=joint_pop_state,
            next_population_state=joint_next_pop_state,
            done=done,
        )

    def _pack_population_states(self, population_states: dict[str, NDArray]) -> NDArray:
        """Pack population states dictionary to single array."""
        packed = []
        for pop_id in sorted(self.population_configs.keys()):
            packed.append(population_states[pop_id])
        return np.concatenate(packed)

    def reset_noise(self) -> None:
        """Reset exploration noise."""
        self.noise.reset()
