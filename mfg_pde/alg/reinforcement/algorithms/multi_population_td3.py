"""
Multi-Population Twin Delayed DDPG (TD3) for Mean Field Games.

Extends TD3 to multiple interacting populations with:
- Twin critics per population: Q₁ᵢ(s,a,m₁,...,mₙ), Q₂ᵢ(s,a,m₁,...,mₙ)
- Delayed policy updates coordinated across populations
- Target policy smoothing per population
- Heterogeneous action spaces supported

Improvements over Multi-Population DDPG:
1. Twin critics reduce overestimation bias in multi-population setting
2. Delayed updates improve stability during Nash equilibrium convergence
3. Target smoothing enhances robustness to population distribution shifts

Mathematical Framework:
- N populations with policies π₁, π₂, ..., πₙ
- Twin Q-functions per population: Q₁ᵢ, Q₂ᵢ for population i
- TD3 target: yᵢ = r + γ·min(Q'₁ᵢ(s',ã',m₁',...,mₙ'), Q'₂ᵢ(s',ã',m₁',...,mₙ'))
  where ã' = μ'ᵢ(s',m₁',...,mₙ') + clip(ε, -c, c)
- Nash equilibrium: Each population best-responds with TD3

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

# Import from multi-population DDPG
from mfg_pde.alg.reinforcement.algorithms.multi_population_ddpg import (
    MultiPopulationDDPGActor,
    MultiPopulationDDPGCritic,
    MultiPopulationReplayBuffer,
)


class MultiPopulationTD3:
    """
    Multi-Population Twin Delayed DDPG for MFG.

    Extends TD3 to handle N interacting populations with:
    - Twin critics per population with cross-population awareness
    - Delayed policy updates coordinated across all populations
    - Target policy smoothing per population
    - Heterogeneous action spaces
    - Nash equilibrium convergence with reduced overestimation

    Key Differences from Multi-Population DDPG:
    - Two critics per population instead of one
    - Clipped double Q-learning: min(Q₁, Q₂) for target
    - Actor updated every d steps (delayed updates)
    - Target action noise for robustness

    Usage:
        env = MultiPopulationMFGEnv(num_populations=3, ...)
        algo = MultiPopulationTD3(
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
        Initialize Multi-Population TD3.

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

        # Default config (TD3-specific parameters)
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,  # Soft update
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "policy_delay": 2,  # TD3: Delayed policy updates
            "target_noise_std": 0.2,  # TD3: Target policy smoothing noise
            "target_noise_clip": 0.5,  # TD3: Clip range for target noise
            "exploration_noise_std": 0.1,  # Exploration noise during training
        }
        self.config = {**default_config, **(config or {})}

        # Initialize networks for each population
        self.actors: list[MultiPopulationDDPGActor] = []
        self.actor_targets: list[MultiPopulationDDPGActor] = []
        # Twin critics per population
        self.critics1: list[MultiPopulationDDPGCritic] = []
        self.critics2: list[MultiPopulationDDPGCritic] = []
        self.critic1_targets: list[MultiPopulationDDPGCritic] = []
        self.critic2_targets: list[MultiPopulationDDPGCritic] = []
        # Optimizers
        self.actor_optimizers: list[Any] = []
        self.critic1_optimizers: list[Any] = []
        self.critic2_optimizers: list[Any] = []
        # Replay buffers
        self.replay_buffers: list[MultiPopulationReplayBuffer] = []

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

            # Twin Critics
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

            # Replay buffer
            replay_buffer = MultiPopulationReplayBuffer(
                capacity=self.config["replay_buffer_size"],
                state_dim=self.state_dims[pop_id],
                action_dim=self.action_dims[pop_id],
                total_pop_dim=total_pop_dim,
            )

            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics1.append(critic1)
            self.critics2.append(critic2)
            self.critic1_targets.append(critic1_target)
            self.critic2_targets.append(critic2_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic1_optimizers.append(critic1_optimizer)
            self.critic2_optimizers.append(critic2_optimizer)
            self.replay_buffers.append(replay_buffer)

        # Training stats
        self.update_count = 0

    def select_actions(
        self, states: dict[int, NDArray], population_states: dict[int, NDArray], training: bool = True
    ) -> dict[int, NDArray]:
        """
        Select actions for all populations with Gaussian exploration noise.

        Args:
            states: {pop_id: state} individual states
            population_states: {pop_id: pop_state} population distributions
            training: If True, add Gaussian exploration noise

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
                # Gaussian noise for exploration (simpler than OU for TD3)
                noise = np.random.normal(0, self.config["exploration_noise_std"], size=action.shape)
                bounds = self.action_bounds[pop_id]
                action = np.clip(action + noise, bounds[0], bounds[1])

            actions[pop_id] = action

        return actions

    def update(self) -> dict[int, tuple[float, float, float]] | None:
        """
        Update all population twin critics and (delayed) actors.

        Returns:
            {pop_id: (critic1_loss, critic2_loss, actor_loss)} or None if buffers too small
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

            # Compute TD3 target with target policy smoothing
            with torch.no_grad():
                # Target actions with smoothing noise
                next_actions = self.actor_targets[pop_id](next_states, next_population_states)

                # Add clipped noise
                noise = torch.randn_like(next_actions) * self.config["target_noise_std"]
                noise = torch.clamp(noise, -self.config["target_noise_clip"], self.config["target_noise_clip"])

                bounds = self.action_bounds[pop_id]
                next_actions_noisy = torch.clamp(next_actions + noise, bounds[0], bounds[1])

                # Clipped double Q-learning: min(Q1, Q2)
                target_q1 = self.critic1_targets[pop_id](next_states, next_actions_noisy, next_population_states)
                target_q2 = self.critic2_targets[pop_id](next_states, next_actions_noisy, next_population_states)
                target_q = torch.min(target_q1, target_q2)

                # TD target
                target = rewards + self.config["discount_factor"] * target_q * (~dones)

            # Update critic 1
            current_q1 = self.critics1[pop_id](states, actions, population_states)
            critic1_loss = nn.functional.mse_loss(current_q1, target)

            self.critic1_optimizers[pop_id].zero_grad()
            critic1_loss.backward()
            self.critic1_optimizers[pop_id].step()

            # Update critic 2
            current_q2 = self.critics2[pop_id](states, actions, population_states)
            critic2_loss = nn.functional.mse_loss(current_q2, target)

            self.critic2_optimizers[pop_id].zero_grad()
            critic2_loss.backward()
            self.critic2_optimizers[pop_id].step()

            # Delayed policy update
            actor_loss = torch.tensor(0.0)
            if self.update_count % self.config["policy_delay"] == 0:
                # Actor loss (use critic1 only, as per TD3 paper)
                actor_actions = self.actors[pop_id](states, population_states)
                actor_loss = -self.critics1[pop_id](states, actor_actions, population_states).mean()

                self.actor_optimizers[pop_id].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[pop_id].step()

                # Soft update target networks
                self._soft_update(self.actors[pop_id], self.actor_targets[pop_id])
                self._soft_update(self.critics1[pop_id], self.critic1_targets[pop_id])
                self._soft_update(self.critics2[pop_id], self.critic2_targets[pop_id])

            losses[pop_id] = (critic1_loss.item(), critic2_loss.item(), actor_loss.item())

        self.update_count += 1
        return losses

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train multi-population TD3 agents.

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
                    for pop_id, (critic1_loss, critic2_loss, actor_loss) in losses.items():
                        critic1_losses[pop_id].append(critic1_loss)
                        critic2_losses[pop_id].append(critic2_loss)
                        if actor_loss > 0:  # Actor was updated
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
                    avg_q1 = np.mean(critic1_losses[pop_id][-100:]) if critic1_losses[pop_id] else 0
                    avg_q2 = np.mean(critic2_losses[pop_id][-100:]) if critic2_losses[pop_id] else 0
                    print(f"  Pop {pop_id}: Reward={avg_reward:.2f}, Q1 Loss={avg_q1:.4f}, Q2 Loss={avg_q2:.4f}")

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic1_losses": critic1_losses,
            "critic2_losses": critic2_losses,
            "actor_losses": actor_losses,
        }
