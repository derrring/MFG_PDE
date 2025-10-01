#!/usr/bin/env python3
"""
MFG Maze Environment Demonstration

Demonstrates the Gymnasium-compatible MFG Maze Environment with:
1. Basic single-agent navigation
2. Population dynamics tracking
3. Different reward structures
4. Various maze types (perfect, recursive division, braided)
5. Multi-episode training simulation

Author: MFG_PDE Team
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

# Check for Gymnasium availability
try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Warning: Gymnasium not available. Install with: pip install gymnasium")

from mfg_pde.alg.reinforcement.environments import (
    ActionType,
    MazeAlgorithm,
    MFGMazeConfig,
    MFGMazeEnvironment,
    PerfectMazeGenerator,
    RecursiveDivisionGenerator,
    RewardType,
    add_loops,
    create_room_based_config,
)


def demo_basic_environment():
    """Demonstrate basic environment usage."""
    if not GYMNASIUM_AVAILABLE:
        print("Skipping demo_basic_environment: Gymnasium not available")
        return

    print("=" * 70)
    print("Demo 1: Basic MFG Maze Environment")
    print("=" * 70)

    # Create simple maze
    generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    generator.generate(seed=42)
    maze_array = generator.to_numpy_array()

    # Configure environment
    config = MFGMazeConfig(
        maze_array=maze_array,
        start_positions=[(1, 1)],
        goal_positions=[(8, 8)],
        action_type=ActionType.FOUR_CONNECTED,
        reward_type=RewardType.SPARSE,
        max_episode_steps=200,
    )

    # Create environment
    env = MFGMazeEnvironment(config, render_mode="human")

    print("\nEnvironment created:")
    print(f"  Maze size: {maze_array.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Run one episode with random policy
    print("\nRunning one episode with random policy...")
    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation['position']}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode terminated at step {step + 1}")
            print(f"  Reason: {'Goal reached' if terminated else 'Max steps'}")
            print(f"  Total reward: {total_reward:.3f}")
            print(f"  Final distance: {info['distance_to_goal']}")
            break

    # Render final state
    print("\nFinal state:")
    env.render()

    env.close()


def demo_population_tracking():
    """Demonstrate population state tracking."""
    if not GYMNASIUM_AVAILABLE:
        print("Skipping demo_population_tracking: Gymnasium not available")
        return

    print("\n" + "=" * 70)
    print("Demo 2: Population State Tracking")
    print("=" * 70)

    # Create maze with open spaces
    config_rd = create_room_based_config(20, 30, room_size="medium", corridor_width="wide", seed=42)
    generator = RecursiveDivisionGenerator(config_rd)
    maze = generator.generate()

    # Configure environment with population observation
    config = MFGMazeConfig(
        maze_array=maze,
        start_positions=[(2, 2), (2, 27), (17, 2), (17, 27)],
        goal_positions=[(10, 15)],
        action_type=ActionType.FOUR_CONNECTED,
        reward_type=RewardType.CONGESTION,
        population_size=100,
        include_population_in_obs=True,
        population_obs_radius=5,
        congestion_weight=0.5,
        max_episode_steps=300,
    )

    env = MFGMazeEnvironment(config)

    print("\nEnvironment with population tracking:")
    print(f"  Population size: {config.population_size}")
    print(f"  Observation radius: {config.population_obs_radius}")
    print(f"  Congestion weight: {config.congestion_weight}")

    # Run episode
    observation, info = env.reset(seed=42)
    print(f"\nLocal density shape: {observation['local_density'].shape}")
    print(f"Initial density sum: {observation['local_density'].sum():.3f}")

    # Simulate a few steps
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: {observation['position']}")
            print(f"  Local density sum: {observation['local_density'].sum():.3f}")
            print(f"  Reward: {reward:.3f}")

        if terminated or truncated:
            break

    env.close()


def demo_reward_structures():
    """Compare different reward structures."""
    if not GYMNASIUM_AVAILABLE:
        print("Skipping demo_reward_structures: Gymnasium not available")
        return

    print("\n" + "=" * 70)
    print("Demo 3: Different Reward Structures")
    print("=" * 70)

    # Create test maze
    generator = PerfectMazeGenerator(15, 15, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    generator.generate(seed=42)
    maze_array = generator.to_numpy_array()

    reward_types = [
        (RewardType.SPARSE, "Sparse (goal only)"),
        (RewardType.DENSE, "Dense (distance shaping)"),
        (RewardType.CONGESTION, "Congestion (population cost)"),
    ]

    for reward_type, description in reward_types:
        print(f"\n{description}:")

        config = MFGMazeConfig(
            maze_array=maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(13, 13)],
            reward_type=reward_type,
            action_type=ActionType.FOUR_CONNECTED,
            max_episode_steps=200,
        )

        env = MFGMazeEnvironment(config)
        observation, info = env.reset(seed=42)

        total_reward = 0
        step_count = 0

        # Run episode with random policy
        for _ in range(200):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        print(f"  Steps: {step_count}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Average reward: {total_reward / step_count:.4f}")
        print(f"  Goal reached: {terminated}")

        env.close()


def demo_maze_types():
    """Compare different maze types."""
    if not GYMNASIUM_AVAILABLE:
        print("Skipping demo_maze_types: Gymnasium not available")
        return

    print("\n" + "=" * 70)
    print("Demo 4: Different Maze Types")
    print("=" * 70)

    maze_configs = []

    # Perfect maze
    generator = PerfectMazeGenerator(20, 20, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    generator.generate(seed=42)
    maze_configs.append(("Perfect Maze", generator.to_numpy_array()))

    # Recursive division
    config_rd = create_room_based_config(20, 20, room_size="medium", corridor_width="medium", seed=42)
    generator_rd = RecursiveDivisionGenerator(config_rd)
    maze_rd = generator_rd.generate()
    maze_configs.append(("Recursive Division", maze_rd))

    # Braided maze
    braided = add_loops(maze_rd, loop_density=0.2, seed=42)
    maze_configs.append(("Braided Maze", braided))

    for maze_name, maze_array in maze_configs:
        print(f"\n{maze_name}:")
        print(f"  Shape: {maze_array.shape}")
        print(f"  Open cells: {np.sum(maze_array == 0)}/{maze_array.size}")
        print(f"  Open ratio: {100 * np.sum(maze_array == 0) / maze_array.size:.1f}%")

        # Create environment
        config = MFGMazeConfig(
            maze_array=maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(18, 18)],
            action_type=ActionType.FOUR_CONNECTED,
            reward_type=RewardType.DENSE,
            max_episode_steps=400,
        )

        env = MFGMazeEnvironment(config)
        observation, info = env.reset(seed=42)

        # Run episode
        steps = 0
        for _ in range(400):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated:
                print(f"  Goal reached in {steps} steps!")
                break

        if not terminated:
            print(f"  Goal not reached in {steps} steps")

        env.close()


def visualize_multi_episode_learning():
    """Visualize performance over multiple episodes."""
    if not GYMNASIUM_AVAILABLE:
        print("Skipping visualize_multi_episode_learning: Gymnasium not available")
        return

    print("\n" + "=" * 70)
    print("Demo 5: Multi-Episode Performance")
    print("=" * 70)

    # Create maze
    generator = PerfectMazeGenerator(15, 15, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    generator.generate(seed=42)
    maze_array = generator.to_numpy_array()

    config = MFGMazeConfig(
        maze_array=maze_array,
        start_positions=[(1, 1)],
        goal_positions=[(13, 13)],
        action_type=ActionType.FOUR_CONNECTED,
        reward_type=RewardType.DENSE,
        max_episode_steps=300,
    )

    env = MFGMazeEnvironment(config)

    # Run multiple episodes with random policy
    num_episodes = 50
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nRunning {num_episodes} episodes with random policy...")

    for episode in range(num_episodes):
        observation, info = env.reset(seed=42 + episode)
        episode_reward = 0
        steps = 0

        for _ in range(300):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated:
                success_count += 1
                break

            if truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(
                f"Episodes {episode - 9:2d}-{episode + 1:2d}: "
                f"Avg reward={avg_reward:6.2f}, Avg length={avg_length:5.1f}"
            )

    print("\nResults:")
    print(f"  Success rate: {100 * success_count / num_episodes:.1f}%")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average length: {np.mean(episode_lengths):.1f}")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(episode_rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Rewards (Random Policy)")
    ax1.grid(True)

    ax2.plot(episode_lengths)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Lengths (Random Policy)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("mfg_maze_learning_curve.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved: mfg_maze_learning_curve.png")
    plt.show()

    env.close()


def main():
    """Run all demonstrations."""
    if not GYMNASIUM_AVAILABLE:
        print("Gymnasium is required for these demonstrations.")
        print("Install with: pip install gymnasium")
        return

    demo_basic_environment()
    demo_population_tracking()
    demo_reward_structures()
    demo_maze_types()
    visualize_multi_episode_learning()

    print("\n" + "=" * 70)
    print("All Demonstrations Complete")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Gymnasium-compatible MFG environment")
    print("  ✓ Population state tracking")
    print("  ✓ Multiple reward structures (sparse, dense, congestion)")
    print("  ✓ Various maze types (perfect, recursive division, braided)")
    print("  ✓ Multi-episode training simulation")
    print("\nNext Steps:")
    print("  - Implement actual RL algorithms (DQN, PPO, etc.)")
    print("  - Add multi-agent population dynamics")
    print("  - Integrate with PDE solvers for hybrid approaches")


if __name__ == "__main__":
    main()
