"""
Mean Field Actor-Critic on Maze Environments.

This example demonstrates training a Mean Field Actor-Critic agent on various
maze environments, showcasing policy gradient methods for MFG problems.

Key Features Demonstrated:
- Actor-Critic training on discrete MFG environments
- Population-aware policy learning
- Comparison across different maze types
- Visualization of learned policies

Mathematical Framework:
- Policy: π(a|s,m) - action probabilities conditioned on state and population
- Value: V(s,m) - expected return conditioned on state and population
- Advantage: A(s,a,m) = Q(s,a,m) - V(s,m)
- Policy Gradient: ∇θ J = E[∇θ log π(a|s,m) A(s,a,m)]

Applications:
- Crowd navigation in complex environments
- Multi-agent coordination in mazes
- Learning equilibrium policies under congestion
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check for PyTorch availability
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
    print("✓ PyTorch available - Actor-Critic can be demonstrated")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - Actor-Critic demo requires PyTorch")
    print("Install with: pip install torch")
    sys.exit(1)

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.algorithms import MeanFieldActorCritic
    from mfg_pde.alg.reinforcement.environments import (
        CellularAutomataGenerator,
        PerfectMazeGenerator,
        RecursiveDivisionGenerator,
        create_museum_hybrid,
    )
    from mfg_pde.alg.reinforcement.environments.mfg_maze_env import (
        MFGMazeConfig,
        MFGMazeEnvironment,
    )

from mfg_pde.utils.logging import configure_research_logging, get_logger  # noqa: E402

# Configure logging
configure_research_logging("actor_critic_maze_demo", level="INFO")
logger = get_logger(__name__)


def create_maze_environment(maze_type: str = "perfect", size: int = 15) -> MFGMazeEnvironment:
    """
    Create maze environment for training.

    Args:
        maze_type: Type of maze (perfect, recursive_division, cellular_automata, hybrid)
        size: Size of maze

    Returns:
        MFG maze environment
    """
    if maze_type == "hybrid":
        # Use museum hybrid preset (Voronoi + CA)
        from mfg_pde.alg.reinforcement.environments.hybrid_maze import HybridMazeGenerator

        hybrid_config = create_museum_hybrid(rows=size, cols=size)
        hybrid_generator = HybridMazeGenerator(hybrid_config)
        maze_array = hybrid_generator.generate()
    elif maze_type == "perfect":
        generator = PerfectMazeGenerator(rows=size, cols=size)
        generator.generate()
        maze_array = generator.to_numpy_array()
    elif maze_type == "recursive_division":
        generator = RecursiveDivisionGenerator(rows=size, cols=size)
        generator.generate()
        maze_array = generator.to_numpy_array()
    elif maze_type == "cellular_automata":
        generator = CellularAutomataGenerator(rows=size, cols=size)
        generator.generate()
        maze_array = generator.to_numpy_array()
    else:
        raise ValueError(f"Unknown maze type: {maze_type}")

    # Create MFG environment config
    config = MFGMazeConfig(
        maze_array=maze_array,
        num_agents=1,  # Single agent for Actor-Critic training
        population_size=100,
        goal_reward=1.0,
        collision_penalty=-1.0,
        max_episode_steps=200,
    )

    env = MFGMazeEnvironment(config=config)
    return env


def train_actor_critic(
    env,
    num_episodes: int = 500,
    max_steps: int = 100,
    device: str = "cpu",
) -> tuple[MeanFieldActorCritic, dict]:
    """
    Train Mean Field Actor-Critic on maze environment.

    Args:
        env: MFG maze environment
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        device: Device to use (cpu/cuda/mps)

    Returns:
        Trained agent and training statistics
    """
    # Initialize agent
    # Observation: position(2) + goal(2) + time(1) + population_density(radius^2)
    # State for agent: position(2) + goal(2) + time(1) = 5
    # Population: local density patch (2*radius+1)^2 = (2*3+1)^2 = 49
    state_dim = 5  # position + goal + time
    action_dim = 4  # 4-connected movement
    radius = 3  # from config.population_obs_radius
    population_dim = (2 * radius + 1) ** 2  # 49

    agent = MeanFieldActorCritic(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        population_dim=population_dim,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        device=device,
    )

    logger.info(f"Training Actor-Critic on {device}")
    logger.info(f"Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    logger.info(f"Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")

    # Train agent
    stats = agent.train(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        log_interval=50,
    )

    return agent, stats


def evaluate_policy(agent, env, num_episodes: int = 10) -> dict:
    """
    Evaluate trained policy.

    Args:
        agent: Trained actor-critic agent
        env: MFG maze environment
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    success_rate = 0

    for episode in range(num_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = agent._extract_state(obs)
        population = agent._extract_population(obs)

        episode_reward = 0
        steps = 0
        max_steps = 200

        for _ in range(max_steps):
            # Use deterministic policy for evaluation
            action, _ = agent.select_action(state, population, deterministic=True)

            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            episode_reward += reward
            steps += 1

            if done:
                if reward > 0:  # Reached goal
                    success_rate += 1
                break

            state = agent._extract_state(next_obs)
            population = agent._extract_population(next_obs)

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_rate / num_episodes,
    }


def plot_training_progress(stats: dict, save_path: str | None = None):
    """
    Plot training statistics.

    Args:
        stats: Training statistics
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot episode rewards
    rewards = np.array(stats["episode_rewards"])
    window = min(20, len(rewards))  # Ensure window doesn't exceed data length
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed_rewards = rewards

    ax1.plot(rewards, alpha=0.3, label="Raw")
    ax1.plot(range(window - 1, len(rewards)), smoothed_rewards, label="Smoothed")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Progress: Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    lengths = np.array(stats["episode_lengths"])
    if len(lengths) >= window:
        smoothed_lengths = np.convolve(lengths, np.ones(window) / window, mode="valid")
    else:
        smoothed_lengths = lengths

    ax2.plot(lengths, alpha=0.3, label="Raw")
    ax2.plot(range(window - 1, len(lengths)), smoothed_lengths, label="Smoothed")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Training Progress: Episode Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training plot saved to {save_path}")

    plt.show()


def compare_maze_types():
    """
    Compare Actor-Critic performance across different maze types.
    """
    logger.info("=" * 60)
    logger.info("Comparing Actor-Critic Across Maze Types")
    logger.info("=" * 60)

    maze_types = ["perfect", "recursive_division", "cellular_automata"]
    results = {}

    for maze_type in maze_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training on {maze_type} maze")
        logger.info(f"{'=' * 60}")

        # Create environment
        env = create_maze_environment(maze_type=maze_type, size=15)

        # Train agent
        agent, stats = train_actor_critic(
            env,
            num_episodes=300,
            max_steps=100,
            device="cpu",
        )

        # Evaluate policy
        eval_stats = evaluate_policy(agent, env, num_episodes=10)

        results[maze_type] = {
            "training": stats,
            "evaluation": eval_stats,
        }

        logger.info(f"\nEvaluation Results ({maze_type}):")
        logger.info(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        logger.info(f"  Mean Length: {eval_stats['mean_length']:.1f}")
        logger.info(f"  Success Rate: {eval_stats['success_rate']:.1%}")

    return results


def main():
    """Main demonstration function."""
    logger.info("=" * 60)
    logger.info("Mean Field Actor-Critic on Maze Environments")
    logger.info("=" * 60)

    # Example 1: Train on a single maze type
    logger.info("\n--- Example 1: Training on Perfect Maze ---")
    env = create_maze_environment(maze_type="perfect", size=15)
    agent, stats = train_actor_critic(env, num_episodes=50, max_steps=50)

    # Evaluate trained policy
    eval_stats = evaluate_policy(agent, env, num_episodes=20)
    logger.info("\nEvaluation Results:")
    logger.info(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    logger.info(f"  Mean Episode Length: {eval_stats['mean_length']:.1f}")
    logger.info(f"  Success Rate: {eval_stats['success_rate']:.1%}")

    # Plot training progress
    plot_training_progress(stats, save_path="actor_critic_training_progress.png")

    # Example 2: Train on hybrid maze (requires larger size for Voronoi)
    logger.info("\n--- Example 2: Training on Hybrid Maze ---")
    hybrid_env = create_maze_environment(maze_type="hybrid", size=35)  # Voronoi requires >=20 per region
    hybrid_agent, hybrid_stats = train_actor_critic(
        hybrid_env,
        num_episodes=50,  # Reduced for demo
        max_steps=50,
    )

    hybrid_eval = evaluate_policy(hybrid_agent, hybrid_env, num_episodes=20)
    logger.info("\nHybrid Maze Evaluation:")
    logger.info(f"  Mean Reward: {hybrid_eval['mean_reward']:.2f} ± {hybrid_eval['std_reward']:.2f}")
    logger.info(f"  Success Rate: {hybrid_eval['success_rate']:.1%}")

    # Example 3: Compare across maze types (optional, takes longer)
    # Uncomment to run full comparison
    # logger.info("\n--- Example 3: Comparison Across Maze Types ---")
    # results = compare_maze_types()

    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)
    logger.info("\nKey Takeaways:")
    logger.info("1. Actor-Critic learns policies conditioned on population state")
    logger.info("2. PPO-style clipping provides stable training")
    logger.info("3. GAE reduces variance in advantage estimation")
    logger.info("4. Learned policies adapt to crowd congestion")


if __name__ == "__main__":
    main()
