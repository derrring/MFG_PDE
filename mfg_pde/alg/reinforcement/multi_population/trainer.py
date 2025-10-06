"""
Training orchestrator for multi-population Mean Field Games.

Coordinates training of heterogeneous populations using different algorithms
(DDPG, TD3, SAC) and monitors Nash equilibrium convergence.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .base_environment import MultiPopulationMFGEnvironment
    from .multi_ddpg import MultiPopulationDDPG
    from .multi_sac import MultiPopulationSAC
    from .multi_td3 import MultiPopulationTD3


class MultiPopulationTrainer:
    """
    Orchestrates training for multi-population MFG.

    Manages:
    - Heterogeneous agents (different algorithms per population)
    - Simultaneous training of all populations
    - Nash equilibrium convergence monitoring
    - Training statistics collection
    """

    def __init__(
        self,
        env: MultiPopulationMFGEnvironment,
        agents: dict[str, MultiPopulationDDPG | MultiPopulationTD3 | MultiPopulationSAC],
    ):
        """
        Initialize multi-population trainer.

        Args:
            env: Multi-population MFG environment
            agents: {pop_id: algorithm_instance}

        Raises:
            ValueError: If agents don't match environment populations
        """
        if set(env.populations.keys()) != set(agents.keys()):
            msg = f"Agent IDs {set(agents.keys())} don't match env populations {set(env.populations.keys())}"
            raise ValueError(msg)

        self.env = env
        self.agents = agents
        self.population_ids = list(agents.keys())

    def train(
        self,
        num_episodes: int = 1000,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> dict[str, Any]:
        """
        Train all populations simultaneously.

        Args:
            num_episodes: Number of training episodes
            verbose: Print training progress
            log_interval: Episodes between logging

        Returns:
            Training statistics per population
        """
        # Initialize statistics
        stats: dict[int, dict[str, list[Any]]] = {
            pop_id: {
                "episode_rewards": [],
                "episode_lengths": [],
                "losses": [],
            }
            for pop_id in self.population_ids
        }

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            population_states = self.env.get_all_population_states()

            # Episode tracking
            episode_rewards = dict.fromkeys(self.population_ids, 0.0)
            episode_lengths = dict.fromkeys(self.population_ids, 0)
            done = dict.fromkeys(self.population_ids, False)

            while not all(done.values()):
                # Select actions for all active populations
                actions = {}
                for pop_id, agent in self.agents.items():
                    if not done[pop_id]:
                        actions[pop_id] = agent.select_action(
                            state=states[pop_id],
                            population_states=population_states,
                            training=True,
                        )
                    else:
                        # Dummy action for terminated population
                        actions[pop_id] = np.zeros(agent.action_dim)

                # Execute joint step
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                next_population_states = self.env.get_all_population_states()

                # Store transitions and update each agent
                for pop_id, agent in self.agents.items():
                    if not done[pop_id]:
                        # Store transition
                        agent.store_transition(
                            state=states[pop_id],
                            action=actions[pop_id],
                            reward=rewards[pop_id],
                            next_state=next_states[pop_id],
                            population_states=population_states,
                            next_population_states=next_population_states,
                            done=terminated[pop_id] or truncated[pop_id],
                        )

                        # Update agent
                        loss = agent.update()
                        if loss is not None:
                            stats[pop_id]["losses"].append(loss)

                        # Accumulate rewards
                        episode_rewards[pop_id] += rewards[pop_id]
                        episode_lengths[pop_id] += 1

                        # Check termination
                        if terminated[pop_id] or truncated[pop_id]:
                            done[pop_id] = True

                # Update state
                states = next_states
                population_states = next_population_states

            # Record episode statistics
            for pop_id in self.population_ids:
                stats[pop_id]["episode_rewards"].append(episode_rewards[pop_id])
                stats[pop_id]["episode_lengths"].append(episode_lengths[pop_id])

            # Reset exploration noise for DDPG agents
            for agent in self.agents.values():
                if hasattr(agent, "reset_noise"):
                    agent.reset_noise()

            # Logging
            if verbose and (episode + 1) % log_interval == 0:
                self._log_progress(episode + 1, num_episodes, stats)

        return stats

    def _log_progress(
        self,
        episode: int,
        total_episodes: int,
        stats: dict[str, Any],
    ) -> None:
        """Print training progress."""
        print(f"\nEpisode {episode}/{total_episodes}")
        print("-" * 80)

        for pop_id in self.population_ids:
            rewards = stats[pop_id]["episode_rewards"]
            lengths = stats[pop_id]["episode_lengths"]

            avg_reward = np.mean(rewards[-100:]) if rewards else 0.0
            avg_length = np.mean(lengths[-100:]) if lengths else 0.0

            # Format loss info based on agent type
            loss_str = "N/A"
            if stats[pop_id]["losses"]:
                last_loss = stats[pop_id]["losses"][-1]
                if isinstance(last_loss, dict):
                    # SAC returns dict
                    loss_str = f"Q1={last_loss['critic1_loss']:.4f}, α={last_loss['alpha']:.3f}"
                elif isinstance(last_loss, tuple):
                    # DDPG/TD3 return tuple
                    if len(last_loss) == 2:
                        loss_str = f"Q={last_loss[0]:.4f}, π={last_loss[1]:.4f}"
                    else:
                        loss_str = f"Q1={last_loss[0]:.4f}, Q2={last_loss[1]:.4f}, π={last_loss[2]:.4f}"

            print(f"  {pop_id:15s} | Reward: {avg_reward:8.2f} | Length: {avg_length:6.1f} | Loss: {loss_str}")

    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate trained policies.

        Args:
            num_episodes: Number of evaluation episodes
            render: Render environment during evaluation

        Returns:
            Evaluation statistics per population
        """
        eval_stats: dict[int, dict[str, list[float]]] = {
            pop_id: {
                "episode_rewards": [],
                "episode_lengths": [],
            }
            for pop_id in self.population_ids
        }

        for _episode in range(num_episodes):
            states, _ = self.env.reset()
            population_states = self.env.get_all_population_states()

            episode_rewards = dict.fromkeys(self.population_ids, 0.0)
            episode_lengths = dict.fromkeys(self.population_ids, 0)
            done = dict.fromkeys(self.population_ids, False)

            while not all(done.values()):
                # Select actions (no exploration)
                actions = {}
                for pop_id, agent in self.agents.items():
                    if not done[pop_id]:
                        actions[pop_id] = agent.select_action(
                            state=states[pop_id],
                            population_states=population_states,
                            training=False,
                        )
                    else:
                        actions[pop_id] = np.zeros(agent.action_dim)

                # Step
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                next_population_states = self.env.get_all_population_states()

                # Update tracking
                for pop_id in self.population_ids:
                    if not done[pop_id]:
                        episode_rewards[pop_id] += rewards[pop_id]
                        episode_lengths[pop_id] += 1

                        if terminated[pop_id] or truncated[pop_id]:
                            done[pop_id] = True

                states = next_states
                population_states = next_population_states

                if render:
                    self.env.render()

            # Record
            for pop_id in self.population_ids:
                eval_stats[pop_id]["episode_rewards"].append(episode_rewards[pop_id])
                eval_stats[pop_id]["episode_lengths"].append(episode_lengths[pop_id])

        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 80)
        for pop_id in self.population_ids:
            rewards = eval_stats[pop_id]["episode_rewards"]
            lengths = eval_stats[pop_id]["episode_lengths"]
            print(
                f"  {pop_id:15s} | Avg Reward: {np.mean(rewards):8.2f} ± {np.std(rewards):6.2f} | "
                f"Avg Length: {np.mean(lengths):6.1f}"
            )

        return eval_stats
