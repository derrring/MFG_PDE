"""
Multi-Population SAC for Mean Field Games.

Extends SAC to heterogeneous multi-population systems with stochastic policies,
entropy regularization, and automatic temperature tuning.

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

from mfg_pde.alg.reinforcement.algorithms.mean_field_ddpg import ReplayBuffer

from .networks import MultiPopulationCritic, MultiPopulationStochasticActor


class MultiPopulationSAC:
    """SAC agent for multi-population MFG with entropy regularization."""

    def __init__(
        self,
        pop_id: str,
        env: MultiPopulationMFGEnvironment,
        population_configs: dict[str, PopulationConfig],
        config: dict[str, Any] | None = None,
    ):
        """Initialize multi-population SAC agent."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.pop_id = pop_id
        self.env = env
        self.population_configs = population_configs
        self.pop_config = population_configs[pop_id]

        # Default config
        default_config = {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "discount_factor": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "replay_buffer_size": 1000000,
            "hidden_dims": [256, 256],
            "target_entropy": None,
            "auto_tune_temperature": True,
            "initial_temperature": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        self.state_dim = self.pop_config.state_dim
        self.action_dim = self.pop_config.action_dim
        self.action_bounds = self.pop_config.action_bounds

        if self.config["target_entropy"] is None:
            self.config["target_entropy"] = -float(self.action_dim)

        # Stochastic actor
        self.actor = MultiPopulationStochasticActor(
            pop_id, self.state_dim, self.action_dim, self.action_bounds, population_configs, self.config["hidden_dims"]
        )

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

        # Temperature
        if self.config["auto_tune_temperature"]:
            self.log_alpha = torch.tensor([np.log(self.config["initial_temperature"])], requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config["alpha_lr"])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config["initial_temperature"]
            self.log_alpha = None

        # Replay buffer
        joint_pop_dim = sum(c.state_dim * 10 for c in population_configs.values())
        self.replay_buffer = ReplayBuffer(
            self.config["replay_buffer_size"], self.state_dim, self.action_dim, joint_pop_dim
        )

    def select_action(self, state: NDArray, population_states: dict[str, NDArray], training: bool = True) -> NDArray:
        """Select action from stochastic policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_states_t = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in population_states.items()}

            if training:
                action, _, _ = self.actor.sample(state_t, pop_states_t)
                return action.squeeze(0).numpy()
            else:
                _, _, mean_action = self.actor.sample(state_t, pop_states_t)
                return mean_action.squeeze(0).numpy()

    def update(self) -> dict[str, float] | None:
        """Update actor, critics, and temperature."""
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

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states, next_pop_dict)
            target_q1 = self.critic1_target(next_states, next_actions, next_pop_dict)
            target_q2 = self.critic2_target(next_states, next_actions, next_pop_dict)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.squeeze(-1)
            target = rewards + self.config["discount_factor"] * target_q * (~dones)

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

        # Update actor
        sampled_actions, log_probs, _ = self.actor.sample(states, pop_dict)
        q1 = self.critic1(states, sampled_actions, pop_dict)
        q2 = self.critic2(states, sampled_actions, pop_dict)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs.squeeze(-1) - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = 0.0
        if self.config["auto_tune_temperature"]:
            alpha_loss = -(
                self.log_alpha.exp() * (log_probs.squeeze(-1) + self.config["target_entropy"]).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        # Soft update targets
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": -log_probs.mean().item(),
        }

    def _unpack_population_states(self, joint_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack joint population states."""
        result = {}
        offset = 0
        for pop_id in sorted(self.population_configs.keys()):
            dim = self.population_configs[pop_id].state_dim * 10
            result[pop_id] = joint_states[:, offset : offset + dim]
            offset += dim
        return result

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
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
