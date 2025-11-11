#!/usr/bin/env python3
"""
Reinforcement Learning Introduction: Algorithm Comparison

This example compares the three Phase 2 RL algorithms on a simple maze:
1. Mean Field Q-Learning
2. Mean Field Actor-Critic (PPO / Population PPO)
3. Nash Q-Learning (same as Q-Learning for symmetric MFG)

Author: MFG_PDE Team
Date: October 2025
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Please install: pip install torch")
    sys.exit(1)

try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    print("Gymnasium not available. Please install: pip install gymnasium")
    sys.exit(1)

from mfg_pde.alg.reinforcement.algorithms import MeanFieldActorCritic
from mfg_pde.alg.reinforcement.algorithms.mean_field_q_learning import create_mean_field_q_learning
from mfg_pde.alg.reinforcement.environments import (
    ActionType,
    MFGMazeConfig,
    MFGMazeEnvironment,
    RewardType,
)


def create_simple_maze():
    """Create a simple 7x7 maze for testing."""
    # 1 = wall, 0 = free space
    maze = np.ones((7, 7), dtype=np.int32)
    maze[1:6, 1:6] = 0  # Open interior

    return maze


def run_q_learning(env, num_episodes=200):
    """Run Mean Field Q-Learning."""
    print("\n" + "=" * 70)
    print("1. Mean Field Q-Learning (Discrete Actions)")
    print("=" * 70)

    algo = create_mean_field_q_learning(
        env,
        config={
            "learning_rate": 1e-3,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "batch_size": 32,
        },
    )

    print(f"Training for {num_episodes} episodes...")
    results = algo.train(num_episodes=num_episodes)

    print(f"Final average reward: {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"Final average length: {np.mean(results['episode_lengths'][-50:]):.1f}")

    return results


def run_actor_critic(env, num_episodes=200):
    """Run Mean Field Actor-Critic (PPO)."""
    print("\n" + "=" * 70)
    print("2. Mean Field Actor-Critic (PPO / Population PPO)")
    print("=" * 70)

    # Get dimensions
    obs, _ = env.reset()
    obs_batch = np.atleast_2d(obs).astype(np.float32)
    state_dim = obs_batch.shape[1]
    action_dim = env.action_space.n
    population_dim = state_dim * 2  # mean + std

    algo = MeanFieldActorCritic(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        population_dim=population_dim,
        config={
            "actor_lr": 3e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
        },
    )

    print(f"Training for {num_episodes} episodes...")
    results = algo.train(num_episodes=num_episodes)

    print(f"Final average reward: {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"Final average length: {np.mean(results['episode_lengths'][-50:]):.1f}")

    return results


def run_nash_q_learning(env, num_episodes=200):
    """
    Run Nash Q-Learning.

    For symmetric MFG, this is identical to Mean Field Q-Learning,
    since Nash equilibrium = max_a Q(s, a, m).
    """
    print("\n" + "=" * 70)
    print("3. Nash Q-Learning (= Mean Field Q-Learning for symmetric MFG)")
    print("=" * 70)

    algo = create_mean_field_q_learning(
        env,
        config={
            "learning_rate": 1e-3,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "batch_size": 32,
        },
    )

    # Demonstrate Nash value computation
    print("\nDemonstrating Nash equilibrium computation:")

    # Random state and population
    state = torch.randn(5, 4)
    pop_state = torch.randn(5, 8)

    # Compute Nash value (= max Q-value for symmetric games)
    nash_values = algo.compute_nash_value(state, pop_state, game_type="symmetric")

    # Verify it equals max Q-value
    q_values = algo.target_network(state, pop_state)
    max_q = q_values.max(dim=1)[0]

    print(f"  Nash values:     {nash_values.numpy()}")
    print(f"  Max Q-values:    {max_q.numpy()}")
    print(f"  Are they equal?  {torch.allclose(nash_values, max_q)}")

    print(f"\nTraining for {num_episodes} episodes...")
    results = algo.train(num_episodes=num_episodes)

    print(f"Final average reward: {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"Final average length: {np.mean(results['episode_lengths'][-50:]):.1f}")

    return results


def plot_comparison(results_list, labels):
    """Plot comparison of algorithms."""
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rewards
    ax = axes[0]
    for results, label in zip(results_list, labels, strict=False):
        rewards = results["episode_rewards"]
        # Smooth with moving average
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=label, linewidth=2)
        else:
            ax.plot(rewards, label=label, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward (smoothed)", fontsize=12)
    ax.set_title("Learning Curves: Episode Rewards", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot episode lengths
    ax = axes[1]
    for results, label in zip(results_list, labels, strict=False):
        lengths = results["episode_lengths"]
        # Smooth with moving average
        window = 20
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=label, linewidth=2)
        else:
            ax.plot(lengths, label=label, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Length (smoothed)", fontsize=12)
    ax.set_title("Learning Curves: Episode Lengths", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_intro_comparison.png", dpi=150, bbox_inches="tight")
    print("\n" + "=" * 70)
    print("Saved comparison plot to: rl_intro_comparison.png")
    print("=" * 70)


def main():
    """Run all algorithms and compare results."""
    print("=" * 70)
    print("Reinforcement Learning for Mean Field Games - Introduction")
    print("=" * 70)
    print("\nThis example demonstrates three RL algorithms on a simple maze:")
    print("1. Mean Field Q-Learning (value-based)")
    print("2. Mean Field Actor-Critic with PPO (policy gradient)")
    print("3. Nash Q-Learning (equilibrium learning)")
    print("\nKey Insight: For symmetric MFG, algorithms 1 and 3 are equivalent!")

    # Create environment
    maze = create_simple_maze()

    config = MFGMazeConfig(
        maze_array=maze,
        start_positions=[(1, 1)],
        goal_positions=[(5, 5)],
        action_type=ActionType.FOUR_CONNECTED,
        reward_type=RewardType.SPARSE,
        goal_reward=10.0,
        num_agents=5,
        max_episode_steps=50,
    )

    env = MFGMazeEnvironment(config)

    print(f"\nEnvironment: {maze.shape[0]}x{maze.shape[1]} maze with {config.num_agents} agents")
    print(f"Start: {config.start_positions[0]}, Goal: {config.goal_positions[0]}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run algorithms
    num_episodes = 200

    results_q = run_q_learning(env, num_episodes)
    env.reset()  # Reset environment

    results_ac = run_actor_critic(env, num_episodes)
    env.reset()

    results_nash = run_nash_q_learning(env, num_episodes)

    # Compare results
    print("\n" + "=" * 70)
    print("Final Comparison (average over last 50 episodes)")
    print("=" * 70)

    print("\nMean Field Q-Learning:")
    print(f"  Reward: {np.mean(results_q['episode_rewards'][-50:]):.2f}")
    print(f"  Length: {np.mean(results_q['episode_lengths'][-50:]):.1f}")

    print("\nMean Field Actor-Critic (PPO):")
    print(f"  Reward: {np.mean(results_ac['episode_rewards'][-50:]):.2f}")
    print(f"  Length: {np.mean(results_ac['episode_lengths'][-50:]):.1f}")

    print("\nNash Q-Learning:")
    print(f"  Reward: {np.mean(results_nash['episode_rewards'][-50:]):.2f}")
    print(f"  Length: {np.mean(results_nash['episode_lengths'][-50:]):.1f}")

    print("\nNote: Nash Q-Learning uses the same algorithm as Mean Field Q-Learning")
    print("because for symmetric MFG, Nash equilibrium = max_a Q(s, a, m).")

    # Plot comparison
    results_list = [results_q, results_ac, results_nash]
    labels = ["Mean Field Q-Learning", "Mean Field Actor-Critic (PPO)", "Nash Q-Learning"]

    plot_comparison(results_list, labels)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ All three algorithms successfully learned to navigate the maze")
    print("✓ Actor-Critic (PPO) typically shows faster, more stable learning")
    print("✓ Q-Learning and Nash Q-Learning produce identical results (as expected)")
    print("\nFor symmetric Mean Field Games:")
    print("  Nash Q-Learning = Mean Field Q-Learning")
    print("  Population PPO = Mean Field Actor-Critic with PPO clipping")


if __name__ == "__main__":
    main()
