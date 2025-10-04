"""
Multi-Population TD3 for Mean Field Games.

Extends TD3 to heterogeneous multi-population systems with twin critics,
delayed policy updates, and target policy smoothing.

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
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import ReplayBuffer

from .networks import MultiPopulationActor, MultiPopulationCritic


class MultiPopulationTD3:
    """TD3 agent for multi-population MFG with twin critics."""

    def __init__(
        self,
        pop_id: str,
        env: MultiPopulationMFGEnvironment,
        population_configs: dict[str, PopulationConfig],
        config: dict[str, Any] | None = None,
    ):
        """Initialize multi-population TD3 agent."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.pop_id = pop_id
        self.env = env
        self.population_configs = population_configs
        self.pop_config = population_configs[pop_id]

        # Default config
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "policy_delay": 2,
            "target_noise_std": 0.2,
            "target_noise_clip": 0.5,
            "exploration_noise_std": 0.1,
        }
        self.config = {**default_config, **(config or {})}

        self.state_dim = self.pop_config.state_dim
        self.action_dim = self.pop_config.action_dim
        self.action_bounds = self.pop_config.action_bounds

        # Actor
        self.actor = MultiPopulationActor(
            pop_id, self.state_dim, self.action_dim, self.action_bounds, population_configs, self.config["hidden_dims"]
        )
        self.actor_target = MultiPopulationActor(
            pop_id, self.state_dim, self.action_dim, self.action_bounds, population_configs, self.config["hidden_dims"]
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics
        self.critic1 = MultiPopulationCritic(
            pop_id, self.state_dim, self.action_dim, population_configs, self.config["hidden_dims"]
        )
        self.critic2 = MultiPopulationCritic(
            pop_id, self.state_dim, self.action_dim, population_configs, self.config["hidden_dims"]
        )
        self.critic1_target = MultiPopulationCritic(
            pop_id, self.state_dim, self.action_dim, population_configs, self.config["hidden_dims"]
        )
        self.critic2_target = MultiPopulationCritic(
            pop_id, self.state_dim, self.action_dim, population_configs, self.config["hidden_dims"]
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config["critic_lr"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config["critic_lr"])

        # Replay buffer
        joint_pop_dim = sum(c.state_dim * 10 for c in population_configs.values())
        self.replay_buffer = ReplayBuffer(
            self.config["replay_buffer_size"], self.state_dim, self.action_dim, joint_pop_dim
        )

        self.update_count = 0

    def select_action(self, state: NDArray, population_states: dict[str, NDArray], training: bool = True) -> NDArray:
        """Select action with Gaussian exploration noise."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_states_t = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in population_states.items()}
            action = self.actor(state_t, pop_states_t).squeeze(0).numpy()

        if training:
            noise = np.random.normal(0, self.config["exploration_noise_std"], size=action.shape)
            action = np.clip(action + noise, self.action_bounds[0], self.action_bounds[1])

        return action

    def update(self) -> tuple[float, float, float] | None:
        """Update twin critics and delayed actor."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None

        batch = self.replay_buffer.sample(self.config["batch_size"])
        states = torch.FloatTensor(batch["states"])
        actions = torch.FloatTensor(batch["actions"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        pop_states = torch.FloatTensor(batch["population_states"])
        next_pop_states = torch.FloatTensor(batch["next_population_states"])
        dones = torch.BoolTensor(batch["dones"])

        pop_dict = self._unpack_population_states(pop_states)
        next_pop_dict = self._unpack_population_states(next_pop_states)

        # TD target with target policy smoothing
        with torch.no_grad():
            next_actions = self.actor_target(next_states, next_pop_dict)
            noise = torch.randn_like(next_actions) * self.config["target_noise_std"]
            noise = torch.clamp(noise, -self.config["target_noise_clip"], self.config["target_noise_clip"])
            next_actions_noisy = torch.clamp(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

            target_q1 = self.critic1_target(next_states, next_actions_noisy, next_pop_dict)
            target_q2 = self.critic2_target(next_states, next_actions_noisy, next_pop_dict)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.config["discount_factor"] * target_q * (~dones)

        # Update critics
        current_q1 = self.critic1(states, actions, pop_dict)
        critic1_loss = nn.functional.mse_loss(current_q1, target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        current_q2 = self.critic2(states, actions, pop_dict)
        critic2_loss = nn.functional.mse_loss(current_q2, target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy update
        actor_loss = torch.tensor(0.0)
        if self.update_count % self.config["policy_delay"] == 0:
            actor_actions = self.actor(states, pop_dict)
            actor_loss = -self.critic1(states, actor_actions, pop_dict).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        self.update_count += 1
        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

    def _unpack_population_states(self, joint_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack joint population states."""
        result = {}
        offset = 0
        for pop_id in sorted(self.population_configs.keys()):
            dim = self.population_configs[pop_id].state_dim * 10
            result[pop_id] = joint_states[:, offset : offset + dim]
            offset += dim
        return result

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
    ):
        """Store transition in replay buffer."""
        joint_pop = np.concatenate([population_states[k] for k in sorted(population_states.keys())])
        joint_next_pop = np.concatenate([next_population_states[k] for k in sorted(next_population_states.keys())])
        self.replay_buffer.push(state, action, reward, next_state, joint_pop, joint_next_pop, done)
