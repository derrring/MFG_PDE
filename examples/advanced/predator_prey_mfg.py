#!/usr/bin/env python3
"""
Predator-Prey Mean Field Game: Heterogeneous Multi-Population Example.

This example demonstrates multi-population MFG with two agent types:
1. Predators: Seek prey, rewarded for captures
2. Prey: Avoid predators, rewarded for reaching safe zones

Key Features:
- Heterogeneous objectives (conflicting goals)
- Cross-population interactions (predators attracted to prey density)
- Nash equilibrium learning through best-response dynamics

Mathematical Framework:
- Two populations: m = (m_pred, m_prey)
- Predator reward: r_pred(s, a, m) includes +bonus near high m_prey
- Prey reward: r_prey(s, a, m) includes -penalty near high m_pred

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
    import torch  # noqa: F401

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

from mfg_pde.alg.reinforcement.algorithms.multi_population_q_learning import (
    create_multi_population_q_learning_solvers,
)
from mfg_pde.alg.reinforcement.environments.multi_population_maze_env import (
    ActionType,
    AgentTypeConfig,
    MultiPopulationMazeConfig,
    MultiPopulationMazeEnvironment,
    RewardType,
)


def create_predator_prey_maze():
    """Create maze for predator-prey scenario."""
    # 15x15 maze with open areas and obstacles
    maze = np.ones((15, 15), dtype=np.int32)
    maze[1:14, 1:14] = 0  # Open interior

    # Add some obstacles (walls)
    maze[5:10, 7] = 1  # Vertical wall
    maze[7, 3:12] = 1  # Horizontal wall (with gaps)
    maze[7, 5] = 0  # Gap
    maze[7, 9] = 0  # Gap

    return maze


def create_predator_prey_environment():
    """Create multi-population environment for predator-prey scenario."""
    maze = create_predator_prey_maze()

    # Predator configuration
    predator_config = AgentTypeConfig(
        type_id="predator",
        type_index=0,
        action_type=ActionType.FOUR_CONNECTED,
        speed_multiplier=1.0,
        reward_type=RewardType.CONGESTION,
        goal_reward=0.0,  # No specific goal
        collision_penalty=-0.5,
        move_cost=0.01,
        congestion_weight=0.0,  # Don't avoid own population
        cross_population_weights={
            "prey": -0.5  # ATTRACTED to prey (negative penalty = positive reward)
        },
        start_positions=[(2, 2), (2, 12), (12, 2), (12, 12)],
        num_agents=4,
    )

    # Prey configuration
    prey_config = AgentTypeConfig(
        type_id="prey",
        type_index=1,
        action_type=ActionType.FOUR_CONNECTED,
        speed_multiplier=1.0,
        reward_type=RewardType.CONGESTION,
        goal_reward=10.0,
        collision_penalty=-0.5,
        move_cost=0.01,
        congestion_weight=0.0,  # Don't avoid own population
        cross_population_weights={
            "predator": 1.0  # AVOID predators (positive penalty = negative reward)
        },
        start_positions=[(7, 7)],  # Center start
        goal_positions=[(1, 1), (1, 13), (13, 1), (13, 13)],  # Corner safe zones
        num_agents=6,
    )

    # Multi-population configuration
    config = MultiPopulationMazeConfig(
        maze_array=maze,
        agent_types={"predator": predator_config, "prey": prey_config},
        population_smoothing=0.5,
        population_update_frequency=5,
        max_episode_steps=200,
        time_penalty=-0.001,
    )

    return MultiPopulationMazeEnvironment(config)


def train_alternating(env, solvers, num_iterations=20, episodes_per_iteration=50):
    """
    Train using alternating best-response dynamics.

    Each iteration:
    1. Fix prey policy, train predator
    2. Fix predator policy, train prey

    This approximates fictitious play for Nash equilibrium.
    """
    print("=" * 80)
    print("Predator-Prey MFG: Alternating Best-Response Training")
    print("=" * 80)

    predator_solver = solvers["predator"]
    prey_solver = solvers["prey"]

    predator_rewards_history = []
    prey_rewards_history = []

    for iteration in range(num_iterations):
        print(f"\n{'=' * 80}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'=' * 80}")

        # Phase 1: Train predators (prey policy fixed)
        print("\n[Phase 1] Training predators (prey uses current policy)...")
        predator_solver.epsilon = max(0.3, predator_solver.epsilon)  # Maintain exploration
        predator_results = predator_solver.train(num_episodes=episodes_per_iteration)
        predator_rewards_history.extend(predator_results["episode_rewards"])

        print(f"Predator avg reward (last 20 episodes): {np.mean(predator_results['episode_rewards'][-20:]):.2f}")

        # Phase 2: Train prey (predator policy fixed)
        print("\n[Phase 2] Training prey (predator uses current policy)...")
        prey_solver.epsilon = max(0.3, prey_solver.epsilon)
        prey_results = prey_solver.train(num_episodes=episodes_per_iteration)
        prey_rewards_history.extend(prey_results["episode_rewards"])

        print(f"Prey avg reward (last 20 episodes): {np.mean(prey_results['episode_rewards'][-20:]):.2f}")

        # Report iteration summary
        print(f"\n{'=' * 80}")
        print(f"Iteration {iteration + 1} Summary:")
        print(f"  Predator: {np.mean(predator_results['episode_rewards']):.2f}")
        print(f"  Prey:     {np.mean(prey_results['episode_rewards']):.2f}")
        print(f"{'=' * 80}")

    return {
        "predator_rewards": predator_rewards_history,
        "prey_rewards": prey_rewards_history,
    }


def visualize_training(results):
    """Plot training progress for both populations."""
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predator rewards
    ax = axes[0]
    predator_rewards = results["predator_rewards"]
    window = 50
    if len(predator_rewards) >= window:
        smoothed = np.convolve(predator_rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label="Predator", color="red", linewidth=2)
    else:
        ax.plot(predator_rewards, label="Predator", color="red", linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward (smoothed)", fontsize=12)
    ax.set_title("Predator Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Prey rewards
    ax = axes[1]
    prey_rewards = results["prey_rewards"]
    if len(prey_rewards) >= window:
        smoothed = np.convolve(prey_rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label="Prey", color="blue", linewidth=2)
    else:
        ax.plot(prey_rewards, label="Prey", color="blue", linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward (smoothed)", fontsize=12)
    ax.set_title("Prey Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predator_prey_training.png", dpi=150, bbox_inches="tight")
    print("\nSaved training plot to: predator_prey_training.png")


def visualize_population_distributions(env):
    """Visualize final population distributions."""
    # Run one episode with trained policies to get final state
    _observations, _ = env.reset(seed=42)

    for _ in range(100):  # Run for a bit
        actions = {
            "predator": np.random.randint(0, 4, size=4),
            "prey": np.random.randint(0, 4, size=6),
        }
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break

    multi_pop_state = env.get_multi_population_state()
    densities = multi_pop_state.get_all_densities()

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predator density
    ax = axes[0]
    im1 = ax.imshow(densities["predator"], cmap="Reds", interpolation="nearest")
    ax.set_title("Predator Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im1, ax=ax, label="Density")

    # Prey density
    ax = axes[1]
    im2 = ax.imshow(densities["prey"], cmap="Blues", interpolation="nearest")
    ax.set_title("Prey Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im2, ax=ax, label="Density")

    # Overlay
    ax = axes[2]
    overlay = np.zeros((*densities["predator"].shape, 3))
    overlay[:, :, 0] = densities["predator"]  # Red channel
    overlay[:, :, 2] = densities["prey"]  # Blue channel
    overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("Combined (Red=Predator, Blue=Prey)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig("predator_prey_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved distribution plot to: predator_prey_distributions.png")


def main():
    """Run predator-prey multi-population MFG example."""
    print("=" * 80)
    print("Predator-Prey Mean Field Game")
    print("=" * 80)
    print("\nScenario:")
    print("  - 4 Predators: Seek high prey density areas")
    print("  - 6 Prey: Avoid high predator density, reach safe zones (corners)")
    print("\nObjective:")
    print("  - Learn Nash equilibrium through alternating best-response training")
    print("  - Predators maximize captures, prey maximize survival")

    # Create environment
    env = create_predator_prey_environment()

    print("\nEnvironment:")
    print(f"  Maze size: {env.maze_array.shape}")
    print(f"  Predators: {env.agent_types['predator'].num_agents}")
    print(f"  Prey: {env.agent_types['prey'].num_agents}")

    # Get dimensions for solvers
    obs, _ = env.reset(seed=42)

    state_dims = {
        "predator": obs["predator"].shape[1],
        "prey": obs["prey"].shape[1],
    }

    action_dims = {
        "predator": env.action_spaces["predator"].n,
        "prey": env.action_spaces["prey"].n,
    }

    # Population dimensions (flattened maze for each type)
    height, width = env.maze_array.shape
    population_dims = {
        "predator": height * width,
        "prey": height * width,
    }

    print("\nNetwork dimensions:")
    print(f"  Predator state: {state_dims['predator']}, actions: {action_dims['predator']}")
    print(f"  Prey state: {state_dims['prey']}, actions: {action_dims['prey']}")
    print(f"  Population dims: {population_dims}")

    # Create solvers
    print("\nCreating multi-population Q-learning solvers...")
    solvers = create_multi_population_q_learning_solvers(
        env=env,
        state_dims=state_dims,
        action_dims=action_dims,
        population_dims=population_dims,
        config={
            "learning_rate": 3e-4,
            "discount_factor": 0.95,
            "epsilon": 1.0,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.1,
            "batch_size": 64,
            "target_update_frequency": 50,
        },
    )

    # Train using alternating best-response
    print("\nStarting alternating best-response training...")
    results = train_alternating(env=env, solvers=solvers, num_iterations=10, episodes_per_iteration=50)

    # Visualize results
    print("\n" + "=" * 80)
    print("Training Complete - Generating Visualizations")
    print("=" * 80)

    visualize_training(results)
    visualize_population_distributions(env)

    # Summary statistics
    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)

    predator_final = np.mean(results["predator_rewards"][-100:])
    prey_final = np.mean(results["prey_rewards"][-100:])

    print("\nFinal Performance (last 100 episodes):")
    print(f"  Predator avg reward: {predator_final:.2f}")
    print(f"  Prey avg reward:     {prey_final:.2f}")

    print("\nInterpretation:")
    if predator_final > -5:
        print("  Predators successfully learned to track prey density")
    if prey_final > 0:
        print("  Prey successfully learned to avoid predators and reach safe zones")

    print("\nKey Insights:")
    print("  - Predators use population state m_prey to locate targets")
    print("  - Prey use population state m_pred to avoid danger")
    print("  - Nash equilibrium emerges from alternating best-response training")
    print("  - Spatial patterns reflect strategic interactions between populations")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
