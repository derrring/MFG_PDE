#!/usr/bin/env python3
"""
Page 45 Maze Experiment with Perfect Maze Generation

Reproduces the experiment from Page 45 using algorithmically generated perfect mazes.
Compares performance across different maze generation algorithms.

Reference: Perfect maze algorithms from "Mazes for Programmers" by Jamis Buck
Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass

from mfg_maze_environment import CellType, MazeConfig, MFGMazeEnvironment
from perfect_maze_generator import (
    MazeAlgorithm,
    PerfectMazeGenerator,
    verify_perfect_maze,
)
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.logging import configure_research_logging, get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for Page 45 maze experiment."""

    maze_rows: int = 15
    maze_cols: int = 15
    num_agents: int = 50
    max_episode_steps: int = 500
    num_episodes: int = 100
    algorithm: MazeAlgorithm = MazeAlgorithm.RECURSIVE_BACKTRACKING
    seed: int | None = None


class Page45MazeExperiment:
    """
    MFG-RL experiment using perfect maze generation.

    Tests agent learning in guaranteed-solvable mazes with different structural properties.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.env: MFGMazeEnvironment | None = None
        self.results: dict = {}

    def setup_environment(self):
        """Create environment with perfect maze."""
        self.logger.info(f"Generating perfect maze: {self.config.maze_rows}x{self.config.maze_cols}")

        # Generate perfect maze
        generator = PerfectMazeGenerator(self.config.maze_rows, self.config.maze_cols, self.config.algorithm)
        grid = generator.generate(seed=self.config.seed)

        # Verify maze is perfect
        verification = verify_perfect_maze(grid)
        self.logger.info(f"Maze verification: {verification}")

        if not verification["is_perfect"]:
            self.logger.warning("Generated maze is not perfect!")

        # Convert to numpy array
        maze_array = generator.to_numpy_array(wall_thickness=1)

        # Create MFG environment configuration
        height, width = maze_array.shape
        maze_config = MazeConfig(
            width=width,
            height=height,
            num_agents=self.config.num_agents,
            max_episode_steps=self.config.max_episode_steps,
            maze_type="custom",
        )

        # Create environment
        self.env = MFGMazeEnvironment(maze_config)

        # Override with perfect maze
        self.env.maze = maze_array.astype(int)

        # Setup start/goal positions
        self._setup_start_goal_positions()

        self.logger.info("Environment setup complete")

    def _setup_start_goal_positions(self):
        """Setup start and goal positions in the perfect maze."""
        # Find all empty cells
        empty_cells = np.argwhere(self.env.maze == 0)

        if len(empty_cells) < 2:
            raise ValueError("Not enough empty cells for start/goal positions")

        # Place start in bottom-left region
        bottom_left = empty_cells[empty_cells[:, 0] > self.env.maze.shape[0] * 0.7]
        bottom_left = bottom_left[bottom_left[:, 1] < self.env.maze.shape[1] * 0.3]

        if len(bottom_left) > 0:
            start_pos = tuple(bottom_left[0])
        else:
            start_pos = tuple(empty_cells[0])

        # Place goal in top-right region
        top_right = empty_cells[empty_cells[:, 0] < self.env.maze.shape[0] * 0.3]
        top_right = top_right[top_right[:, 1] > self.env.maze.shape[1] * 0.7]

        if len(top_right) > 0:
            goal_pos = tuple(top_right[0])
        else:
            goal_pos = tuple(empty_cells[-1])

        # Set positions
        self.env.maze[start_pos] = CellType.START.value
        self.env.maze[goal_pos] = CellType.GOAL.value

        self.env.start_position = start_pos
        self.env.goal_position = goal_pos

        self.logger.info(f"Start: {start_pos}, Goal: {goal_pos}")

    def run_experiment(self) -> dict:
        """Run the full experiment."""
        if self.env is None:
            self.setup_environment()

        self.logger.info(f"Running {self.config.num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        success_count = 0
        goal_reached_steps = []

        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            _obs = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < self.config.max_episode_steps:
                # Simple random policy for baseline
                actions = np.random.randint(0, 5, size=self.config.num_agents)

                _obs, rewards, done, info = self.env.step(actions)

                episode_reward += np.mean(rewards)
                steps += 1

                # Check if any agent reached goal
                if info.get("goal_reached", False):
                    success_count += 1
                    goal_reached_steps.append(steps)
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

        # Compile results
        self.results = {
            "algorithm": self.config.algorithm.value,
            "maze_size": (self.config.maze_rows, self.config.maze_cols),
            "num_agents": self.config.num_agents,
            "num_episodes": self.config.num_episodes,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "success_rate": success_count / self.config.num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "avg_success_steps": np.mean(goal_reached_steps) if goal_reached_steps else float("inf"),
        }

        self.logger.info(f"Experiment complete: Success rate = {self.results['success_rate']:.2%}")

        return self.results

    def visualize_results(self):
        """Visualize experiment results."""
        if not self.results:
            self.logger.warning("No results to visualize")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Maze visualization
        ax_maze = fig.add_subplot(gs[0:2, 0])
        ax_maze.imshow(self.env.maze, cmap="binary", interpolation="nearest")
        ax_maze.set_title(f"Perfect Maze: {self.config.algorithm.value}", fontweight="bold")
        ax_maze.set_xticks([])
        ax_maze.set_yticks([])

        # Mark start and goal
        start = self.env.start_position
        goal = self.env.goal_position
        ax_maze.plot(start[1], start[0], "go", markersize=15, label="Start")
        ax_maze.plot(goal[1], goal[0], "r*", markersize=20, label="Goal")
        ax_maze.legend(loc="upper right")

        # Episode rewards
        ax_rewards = fig.add_subplot(gs[0, 1:])
        rewards = self.results["episode_rewards"]
        ax_rewards.plot(rewards, alpha=0.6, linewidth=1)
        ax_rewards.plot(np.convolve(rewards, np.ones(10) / 10, mode="valid"), linewidth=2, label="Moving Average (10)")
        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Cumulative Reward")
        ax_rewards.set_title("Learning Progress: Episode Rewards")
        ax_rewards.grid(True, alpha=0.3)
        ax_rewards.legend()

        # Episode lengths
        ax_lengths = fig.add_subplot(gs[1, 1:])
        lengths = self.results["episode_lengths"]
        ax_lengths.plot(lengths, alpha=0.6, linewidth=1)
        ax_lengths.plot(np.convolve(lengths, np.ones(10) / 10, mode="valid"), linewidth=2, label="Moving Average (10)")
        ax_lengths.set_xlabel("Episode")
        ax_lengths.set_ylabel("Steps to Goal/Timeout")
        ax_lengths.set_title("Episode Lengths")
        ax_lengths.grid(True, alpha=0.3)
        ax_lengths.legend()

        # Statistics
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis("off")

        stats_text = f"""
Experiment Statistics (Algorithm: {self.results['algorithm']})
{'=' * 70}

Maze Configuration:
  Size: {self.results['maze_size']}
  Number of Agents: {self.results['num_agents']}
  Episodes: {self.results['num_episodes']}

Performance Metrics:
  Success Rate: {self.results['success_rate']:.2%}
  Average Reward: {self.results['avg_reward']:.2f}
  Average Episode Length: {self.results['avg_episode_length']:.1f} steps
  Average Steps to Success: {self.results['avg_success_steps']:.1f} steps

Maze Properties:
  Algorithm: {self.config.algorithm.value}
  Guaranteed Properties:
    - Fully Connected: Every cell reachable from every other cell
    - No Loops: Exactly one path between any two cells
    - Perfect Maze: Minimal spanning tree structure
"""

        ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
        )

        fig.suptitle("Page 45 Experiment: Perfect Maze MFG-RL", fontsize=16, fontweight="bold")

        plt.tight_layout()
        return fig


def compare_maze_algorithms():
    """Compare different maze generation algorithms for MFG-RL."""
    print("Comparing Maze Generation Algorithms for MFG-RL")
    print("=" * 70)

    algorithms = [
        MazeAlgorithm.RECURSIVE_BACKTRACKING,
        MazeAlgorithm.BINARY_TREE,
        MazeAlgorithm.SIDEWINDER,
        MazeAlgorithm.WILSONS,
    ]

    results = []

    for algorithm in algorithms:
        print(f"\nTesting {algorithm.value}...")

        config = ExperimentConfig(
            maze_rows=15,
            maze_cols=15,
            num_agents=50,
            max_episode_steps=300,
            num_episodes=50,
            algorithm=algorithm,
            seed=42,
        )

        experiment = Page45MazeExperiment(config)
        result = experiment.run_experiment()
        results.append(result)

        print(f"  Success Rate: {result['success_rate']:.2%}")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")

    # Comparative visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, (algorithm, result) in enumerate(zip(algorithms, results, strict=False)):
        ax = axes[i // 2, i % 2]

        rewards = result["episode_rewards"]
        ax.plot(rewards, alpha=0.6, linewidth=1)
        ax.plot(np.convolve(rewards, np.ones(5) / 5, mode="valid"), linewidth=2, label="Moving Avg")

        ax.set_title(f"{algorithm.value}\nSuccess: {result['success_rate']:.1%}", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Algorithm Comparison: MFG-RL Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("Comparison Summary:")
    print(f"{'Algorithm':<25} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 55)
    for result in results:
        print(f"{result['algorithm']:<25} {result['success_rate']:>12.2%}   {result['avg_reward']:>12.2f}")


def main():
    """Run Page 45 perfect maze experiment."""
    configure_research_logging("page45_perfect_maze", level="INFO")

    print("Page 45 Maze Experiment with Perfect Maze Generation")
    print("=" * 70)
    print("Reference: 'Mazes for Programmers' by Jamis Buck")
    print()

    # Single algorithm detailed experiment
    config = ExperimentConfig(
        maze_rows=20,
        maze_cols=20,
        num_agents=50,
        max_episode_steps=500,
        num_episodes=100,
        algorithm=MazeAlgorithm.RECURSIVE_BACKTRACKING,
        seed=42,
    )

    experiment = Page45MazeExperiment(config)
    experiment.run_experiment()
    experiment.visualize_results()
    plt.show()

    # Comparative analysis
    print("\nRunning comparative analysis across algorithms...")
    compare_maze_algorithms()


if __name__ == "__main__":
    main()
