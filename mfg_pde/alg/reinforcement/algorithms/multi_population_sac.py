"""
Multi-Population Soft Actor-Critic (SAC) for Mean Field Games.

Extends SAC to multiple interacting populations with:
- Stochastic policies per population: πᵢ(a|s, m₁, ..., mₙ)
- Twin soft Q-critics per population with cross-population awareness
- Per-population temperature tuning for entropy regularization
- Maximum entropy objective with Nash equilibrium convergence

Key Features:
1. Entropy-regularized policies encourage exploration in multi-population setting
2. Automatic temperature tuning per population for balanced learning
3. Reparameterization trick enables gradient flow through stochastic sampling
4. Robust to population distribution shifts via maximum entropy

Mathematical Framework:
- Objective: Jᵢ(πᵢ) = E[Σ γᵗ(rᵢ(sₜ,aₜ,m₁,...,mₙ) + αᵢ𝓗(πᵢ(·|sₜ,m₁,...,mₙ)))]
- Soft Bellman: Qᵢ(s,a,m) = r + γ E[min(Q₁ᵢ',Q₂ᵢ')(s',a',m') - αᵢ log πᵢ(a'|s',m')]
- Policy: πᵢ(a|s,m) = tanh(𝒩(μθᵢ(s,m₁,...,mₙ), σθᵢ(s,m₁,...,mₙ)))

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
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Normal = None

# Import from multi-population implementations
from mfg_pde.alg.reinforcement.algorithms.multi_population_ddpg import (
    MultiPopulationDDPGCritic,
    MultiPopulationReplayBuffer,
)

# Constants for numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MultiPopulationSACStochasticActor(nn.Module):
    """
    Stochastic actor network for one population in multi-population SAC.

    Architecture:
    - State encoder: s → features
    - Multi-population encoder: (m₁, m₂, ..., mₙ) → features
    - Mean head: features → μ(s, m₁, ..., mₙ)
    - Log std head: features → log σ(s, m₁, ..., mₙ)

    Outputs Gaussian policy with tanh squashing for bounded actions.
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
        Initialize multi-population SAC stochastic actor.

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
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_pop_dim = sum(population_dims)

        # Shared encoder for state + all populations
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + self.total_pop_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dims[1], action_dim)

        # Log std head
        self.log_std_head = nn.Linear(hidden_dims[1], action_dim)

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state: torch.Tensor, population_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute mean and log_std.

        Args:
            state: Individual state [batch, state_dim]
            population_states: Concatenated population states [batch, total_pop_dim]

        Returns:
            (mean, log_std) tuple
        """
        features = self.encoder(torch.cat([state, population_states], dim=1))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self, state: torch.Tensor, population_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Args:
            state: Individual state [batch, state_dim]
            population_states: Concatenated population states [batch, total_pop_dim]

        Returns:
            (action, log_prob, mean_action) tuple
        """
        mean, log_std = self.forward(state, population_states)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradient

        # Squash to action bounds with tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Compute log probability with change of variables
        log_prob = normal.log_prob(x_t)
        # Correction for tanh squashing: log π(a) = log μ(ã) - Σ log(1 - tanh²(ã))
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Also return mean for evaluation
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class MultiPopulationSAC:
    """
    Multi-Population Soft Actor-Critic for MFG.

    Extends SAC to handle N interacting populations with:
    - Stochastic policies per population with entropy regularization
    - Twin soft Q-critics per population with cross-population awareness
    - Per-population automatic temperature tuning
    - Maximum entropy objective
    - Nash equilibrium convergence with exploration

    Key Advantages over Multi-Population TD3:
    - Stochastic policies naturally explore multiple equilibria
    - Automatic temperature tuning balances exploitation/exploration
    - More robust to population distribution shifts
    - Better sample efficiency in multi-population settings

    Usage:
        env = MultiPopulationMFGEnv(num_populations=3, ...)
        algo = MultiPopulationSAC(
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
        Initialize Multi-Population SAC.

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

        # Default config (SAC-specific parameters)
        default_config = {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "discount_factor": 0.99,
            "tau": 0.005,  # Soft update (SAC uses slightly higher than TD3)
            "batch_size": 256,
            "replay_buffer_size": 1000000,
            "hidden_dims": [256, 256],
            "target_entropy": None,  # Auto-computed per population as -action_dim
            "auto_tune_temperature": True,
            "initial_temperature": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        # Initialize networks for each population
        self.actors: list[MultiPopulationSACStochasticActor] = []
        # Twin critics per population
        self.critics1: list[MultiPopulationDDPGCritic] = []
        self.critics2: list[MultiPopulationDDPGCritic] = []
        self.critic1_targets: list[MultiPopulationDDPGCritic] = []
        self.critic2_targets: list[MultiPopulationDDPGCritic] = []
        # Optimizers
        self.actor_optimizers: list[Any] = []
        self.critic1_optimizers: list[Any] = []
        self.critic2_optimizers: list[Any] = []
        # Temperature parameters (per population)
        self.log_alphas: list[torch.Tensor | None] = []
        self.alpha_optimizers: list[Any | None] = []
        self.alphas: list[float] = []
        self.target_entropies: list[float] = []
        # Replay buffers
        self.replay_buffers: list[MultiPopulationReplayBuffer] = []

        total_pop_dim = sum(self.population_dims)

        for pop_id in range(num_populations):
            # Stochastic Actor
            actor = MultiPopulationSACStochasticActor(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                action_bounds=self.action_bounds[pop_id],
                hidden_dims=self.config["hidden_dims"],
            )

            # Twin Critics (reuse DDPG critic architecture)
            critic1 = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )
            critic2 = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )

            critic1_target = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )
            critic2_target = MultiPopulationDDPGCritic(
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                population_dims=self.population_dims,
                hidden_dims=self.config["hidden_dims"],
            )

            critic1_target.load_state_dict(critic1.state_dict())
            critic2_target.load_state_dict(critic2.state_dict())

            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.config["actor_lr"])
            critic1_optimizer = optim.Adam(critic1.parameters(), lr=self.config["critic_lr"])
            critic2_optimizer = optim.Adam(critic2.parameters(), lr=self.config["critic_lr"])

            # Temperature (entropy weight) - per population
            target_entropy = self.config["target_entropy"]
            if target_entropy is None:
                target_entropy = -float(self.action_dims[pop_id])

            if self.config["auto_tune_temperature"]:
                log_alpha = torch.tensor([np.log(self.config["initial_temperature"])], requires_grad=True)
                alpha_optimizer = optim.Adam([log_alpha], lr=self.config["alpha_lr"])
                alpha = log_alpha.exp().item()
            else:
                log_alpha = None
                alpha_optimizer = None
                alpha = self.config["initial_temperature"]

            # Replay buffer
            replay_buffer = MultiPopulationReplayBuffer(
                capacity=self.config["replay_buffer_size"],
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                total_pop_dim=total_pop_dim,
            )

            self.actors.append(actor)
            self.critics1.append(critic1)
            self.critics2.append(critic2)
            self.critic1_targets.append(critic1_target)
            self.critic2_targets.append(critic2_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic1_optimizers.append(critic1_optimizer)
            self.critic2_optimizers.append(critic2_optimizer)
            self.log_alphas.append(log_alpha)
            self.alpha_optimizers.append(alpha_optimizer)
            self.alphas.append(alpha)
            self.target_entropies.append(target_entropy)
            self.replay_buffers.append(replay_buffer)

        # Training stats
        self.update_count = 0

    def select_actions(
        self, states: dict[int, NDArray], population_states: dict[int, NDArray], training: bool = True
    ) -> dict[int, NDArray]:
        """
        Select actions for all populations using stochastic policies.

        Args:
            states: {pop_id: state} individual states
            population_states: {pop_id: pop_state} population distributions
            training: If True, sample from policy; else use mean

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

                if training:
                    action, _, _ = self.actors[pop_id].sample(state_t, pop_state_t)
                    actions[pop_id] = action.squeeze(0).numpy()
                else:
                    # Use mean for evaluation
                    _, _, mean_action = self.actors[pop_id].sample(state_t, pop_state_t)
                    actions[pop_id] = mean_action.squeeze(0).numpy()

        return actions

    def update(self) -> dict[int, dict[str, float]] | None:
        """
        Update all population actors, twin critics, and temperatures.

        Returns:
            {pop_id: {'critic1_loss': ..., 'critic2_loss': ..., 'actor_loss': ..., 'alpha_loss': ..., 'alpha': ...}}
            or None if buffers too small
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

            # Update critics
            with torch.no_grad():
                # Sample next actions from current policy
                next_actions, next_log_probs, _ = self.actors[pop_id].sample(next_states, next_population_states)

                # Compute soft target: min(Q1', Q2') - α log π
                target_q1 = self.critic1_targets[pop_id](next_states, next_actions, next_population_states)
                target_q2 = self.critic2_targets[pop_id](next_states, next_actions, next_population_states)
                target_q = torch.min(target_q1, target_q2) - self.alphas[pop_id] * next_log_probs.squeeze(-1)

                # Soft Bellman backup
                target = rewards + self.config["discount_factor"] * target_q * (~dones)

            # Critic 1 loss
            current_q1 = self.critics1[pop_id](states, actions, population_states)
            critic1_loss = nn.functional.mse_loss(current_q1, target)

            self.critic1_optimizers[pop_id].zero_grad()
            critic1_loss.backward()
            self.critic1_optimizers[pop_id].step()

            # Critic 2 loss
            current_q2 = self.critics2[pop_id](states, actions, population_states)
            critic2_loss = nn.functional.mse_loss(current_q2, target)

            self.critic2_optimizers[pop_id].zero_grad()
            critic2_loss.backward()
            self.critic2_optimizers[pop_id].step()

            # Update actor
            sampled_actions, log_probs, _ = self.actors[pop_id].sample(states, population_states)

            q1_new = self.critics1[pop_id](states, sampled_actions, population_states)
            q2_new = self.critics2[pop_id](states, sampled_actions, population_states)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - α log π (minimize negative)
            actor_loss = (self.alphas[pop_id] * log_probs.squeeze(-1) - q_new).mean()

            self.actor_optimizers[pop_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[pop_id].step()

            # Update temperature (if auto-tuning)
            alpha_loss = torch.tensor(0.0)
            if self.config["auto_tune_temperature"]:
                # Temperature loss: α * (-log π - target_entropy)
                alpha_loss = (
                    self.log_alphas[pop_id] * (-log_probs.squeeze(-1).detach() - self.target_entropies[pop_id])
                ).mean()

                self.alpha_optimizers[pop_id].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizers[pop_id].step()

                # Update alpha value
                self.alphas[pop_id] = self.log_alphas[pop_id].exp().item()

            # Soft update target networks
            self._soft_update(self.critics1[pop_id], self.critic1_targets[pop_id])
            self._soft_update(self.critics2[pop_id], self.critic2_targets[pop_id])

            losses[pop_id] = {
                "critic1_loss": critic1_loss.item(),
                "critic2_loss": critic2_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alphas[pop_id],
            }

        self.update_count += 1
        return losses

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train multi-population SAC agents.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training statistics per population
        """
        # Per-population tracking
        episode_rewards: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        critic1_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        critic2_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        actor_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        alpha_losses: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}
        alphas: dict[int, list[float]] = {i: [] for i in range(self.num_populations)}

        episode_lengths = []

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward_dict = dict.fromkeys(range(self.num_populations), 0.0)
            episode_length = 0

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
                    for pop_id, loss_dict in losses.items():
                        critic1_losses[pop_id].append(loss_dict["critic1_loss"])
                        critic2_losses[pop_id].append(loss_dict["critic2_loss"])
                        actor_losses[pop_id].append(loss_dict["actor_loss"])
                        alpha_losses[pop_id].append(loss_dict["alpha_loss"])
                        alphas[pop_id].append(loss_dict["alpha"])

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
                    avg_q1 = np.mean(critic1_losses[pop_id][-100:]) if critic1_losses[pop_id] else 0
                    avg_q2 = np.mean(critic2_losses[pop_id][-100:]) if critic2_losses[pop_id] else 0
                    avg_alpha = np.mean(alphas[pop_id][-100:]) if alphas[pop_id] else self.alphas[pop_id]
                    print(
                        f"  Pop {pop_id}: Reward={avg_reward:.2f}, "
                        f"Q1 Loss={avg_q1:.4f}, Q2 Loss={avg_q2:.4f}, "
                        f"Alpha={avg_alpha:.4f}"
                    )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic1_losses": critic1_losses,
            "critic2_losses": critic2_losses,
            "actor_losses": actor_losses,
            "alpha_losses": alpha_losses,
            "alphas": alphas,
        }
