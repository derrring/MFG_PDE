#!/usr/bin/env python3
"""
Page 45 Paper-Style Maze Demo

This script demonstrates the MFG maze environment using a layout inspired by
research papers, specifically designed to test mean field interactions through
spatial bottlenecks and congestion dynamics.

Features:
- Paper-style labyrinth configuration
- Real-time visualization of agent movement and congestion
- Multiple policy demonstrations
- Interactive controls for stepping through episodes
- Analysis of mean field effects

Usage:
    python demo_page45_maze.py --interactive
    python demo_page45_maze.py --policy greedy --agents 100
    python demo_page45_maze.py --analysis

Author: MFG_PDE Team
Date: October 2025
"""

import argparse

# Import our maze modules
from mfg_maze_layouts import CustomMazeLoader, MazeAnalyzer, create_custom_maze_environment
from test_mfg_maze_comprehensive import CongestionAwarePolicy, GreedyPolicy, RandomPolicy, ShortestPathPolicy

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.logging import configure_research_logging, get_logger

logger = get_logger(__name__)


class InteractiveMazeDemo:
    """Interactive demo for the page 45 maze layout."""

    def __init__(self, num_agents: int = 50, policy_name: str = "greedy"):
        self.logger = get_logger(__name__)

        # Setup logging
        configure_research_logging("page45_maze_demo", level="INFO")

        # Create environment with page 45 layout
        self.env = create_custom_maze_environment("paper_page45", num_agents, max_steps=1000)

        # Setup policy
        self.policy = self._create_policy(policy_name)

        # Demo state
        self.current_step = 0
        self.episode_history = []
        self.congestion_history = []

        # Visualization
        self.fig = None
        self.ax = None
        self.animation = None

        self.logger.info(f"Created Page 45 Maze Demo with {num_agents} agents using {policy_name} policy")

    def _create_policy(self, policy_name: str):
        """Create policy based on name."""
        policies = {
            "random": RandomPolicy(),
            "greedy": GreedyPolicy(),
            "congestion": CongestionAwarePolicy(1.0),
            "shortest": ShortestPathPolicy(),
        }

        if policy_name not in policies:
            self.logger.warning(f"Unknown policy {policy_name}, using greedy")
            policy_name = "greedy"

        return policies[policy_name]

    def run_episode(self, max_steps: int = 500, visualize: bool = True) -> dict:
        """Run a single episode with optional visualization."""
        self.logger.info(f"Running episode with {self.policy.name} policy")

        # Reset environment
        obs = self.env.reset()

        # Episode tracking
        episode_data = {
            "steps": [],
            "rewards": [],
            "congestion": [],
            "agents_at_goal": [],
            "mean_distance": [],
        }

        if visualize:
            self.env.render()
            plt.pause(1.0)  # Initial pause to see starting state

        for step in range(max_steps):
            # Get actions from policy
            actions = np.zeros(self.env.config.num_agents, dtype=int)
            for i in range(self.env.config.num_agents):
                if not self.env.agents_reached_goal[i]:
                    actions[i] = self.policy.get_action(i, obs[i], self.env)

            # Execute step
            obs, rewards, done, info = self.env.step(actions)

            # Track episode data
            episode_data["steps"].append(step)
            episode_data["rewards"].append(np.mean(rewards))
            episode_data["congestion"].append(np.mean(info["congestion_map"]))
            episode_data["agents_at_goal"].append(info["agents_at_goal"])
            episode_data["mean_distance"].append(info["mean_distance_to_goal"])

            if visualize:
                self.env.render()
                plt.pause(0.1)

                # Print step info
                if step % 50 == 0:
                    print(
                        f"Step {step}: {info['agents_at_goal']}/{self.env.config.num_agents} "
                        f"at goal, avg distance: {info['mean_distance_to_goal']:.1f}"
                    )

            if done:
                break

        # Final statistics
        episode_summary = {
            "total_steps": len(episode_data["steps"]),
            "success_rate": episode_data["agents_at_goal"][-1] / self.env.config.num_agents,
            "total_reward": sum(episode_data["rewards"]),
            "avg_congestion": np.mean(episode_data["congestion"]),
            "final_distance": episode_data["mean_distance"][-1],
            "policy": self.policy.name,
            "episode_data": episode_data,
        }

        self.logger.info(
            f"Episode completed: {episode_summary['success_rate']:.3f} success rate, "
            f"{episode_summary['total_steps']} steps"
        )

        return episode_summary

    def analyze_maze_properties(self):
        """Analyze the page 45 maze layout properties."""
        print("\n🔍 Page 45 Maze Analysis")
        print("=" * 40)

        # Load maze for analysis
        loader = CustomMazeLoader()
        maze = loader.get_predefined_layout("paper_page45")
        analyzer = MazeAnalyzer(maze)

        # Get comprehensive statistics
        stats = analyzer.get_maze_statistics()

        print(f"📐 Dimensions: {stats['dimensions']}")
        print(f"🧱 Wall Density: {stats['wall_density']:.3f}")
        print(f"🔗 Connected Components: {stats['connectivity']['connected_components']}")
        print(f"🚪 Total Bottlenecks: {len(stats['bottlenecks']['bottlenecks'])}")
        print(f"⚠️  Critical Bottlenecks: {len(stats['bottlenecks']['critical_bottlenecks'])}")

        # Test paths between corners
        empty_cells = analyzer._get_empty_cells()

        # Find corner cells
        corners = {
            "top_left": min(empty_cells, key=lambda p: p[0] + p[1]),
            "top_right": min(empty_cells, key=lambda p: p[0] - p[1]),
            "bottom_left": min(empty_cells, key=lambda p: -p[0] + p[1]),
            "bottom_right": min(empty_cells, key=lambda p: -p[0] - p[1]),
        }

        print("\n🗺️  Path Analysis:")
        for corner_name, corner_pos in corners.items():
            print(f"   {corner_name}: {corner_pos}")

        # Test path from top-left to bottom-right
        start = corners["top_left"]
        goal = corners["bottom_right"]
        path_metrics = analyzer.compute_path_metrics(start, goal)

        if path_metrics["path_exists"]:
            print("\n📏 Diagonal Path (TL→BR):")
            print(f"   Manhattan Distance: {path_metrics['manhattan_distance']}")
            print(f"   Shortest Path: {path_metrics['shortest_path_length']} steps")
            print(f"   Detour Ratio: {path_metrics['detour_ratio']:.2f}")
        else:
            print(f"\n❌ No path exists from {start} to {goal}")

        # Visualize analysis
        self._visualize_maze_analysis(maze, analyzer, stats)

    def _visualize_maze_analysis(self, maze: np.ndarray, analyzer: MazeAnalyzer, stats: dict):
        """Visualize maze analysis results."""
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Original maze
        ax1.imshow(maze, cmap="binary", interpolation="nearest")
        ax1.set_title("Page 45 Maze Layout")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Bottlenecks visualization
        bottleneck_maze = maze.copy().astype(float)
        bottlenecks_info = analyzer.analyze_bottlenecks()

        for bottleneck in bottlenecks_info["bottlenecks"]:
            bottleneck_maze[bottleneck] = 0.5

        for critical in bottlenecks_info["critical_bottlenecks"]:
            bottleneck_maze[critical] = 0.2

        ax2.imshow(bottleneck_maze, cmap="RdYlBu_r", interpolation="nearest")
        ax2.set_title("Bottlenecks (Red = Critical)")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Distance heatmap from center
        height, width = maze.shape
        center = (height // 2, width // 2)
        distance_map = np.full_like(maze, float("inf"), dtype=float)

        for i in range(height):
            for j in range(width):
                if maze[i, j] != 0:  # Not a wall
                    distance_map[i, j] = abs(i - center[0]) + abs(j - center[1])

        distance_map[maze == 0] = np.nan  # Walls as NaN

        im3 = ax3.imshow(distance_map, cmap="viridis", interpolation="nearest")
        ax3.set_title("Distance from Center")
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3)

        # Statistics summary
        stats_text = f"""
Page 45 Maze Statistics:

Structural Properties:
• Dimensions: {stats['dimensions']}
• Total Cells: {stats['total_cells']}
• Wall Density: {stats['wall_density']:.1%}
• Empty Cells: {stats['empty_cells']}

Connectivity Analysis:
• Components: {stats['connectivity']['connected_components']}
• Largest Component: {stats['connectivity']['largest_component_size']} cells
• Fully Connected: {stats['connectivity']['is_fully_connected']}

Bottleneck Analysis:
• Total Bottlenecks: {len(stats['bottlenecks']['bottlenecks'])}
• Critical Bottlenecks: {len(stats['bottlenecks']['critical_bottlenecks'])}
• Bottleneck Density: {stats['bottlenecks']['bottleneck_density']:.1%}

Mean Field Properties:
• High congestion expected at bottlenecks
• Strategic positioning matters
• Multiple paths create routing choices
"""

        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

        plt.tight_layout()
        plt.suptitle("Page 45 Maze: Mean Field Games Analysis", fontsize=16, y=1.02)
        plt.show()

    def compare_policies(self):
        """Compare different policies on the page 45 maze."""
        print("\n🏁 Policy Comparison on Page 45 Maze")
        print("=" * 45)

        policies = [
            ("Random", RandomPolicy()),
            ("Greedy", GreedyPolicy()),
            ("Congestion-Aware", CongestionAwarePolicy(1.0)),
            ("Shortest Path", ShortestPathPolicy()),
        ]

        results = {}

        for policy_name, policy in policies:
            print(f"\n🤖 Testing {policy_name} Policy...")

            # Run multiple episodes
            episode_results = []
            for episode in range(3):
                # Create fresh environment
                env = create_custom_maze_environment("paper_page45", 50, 300)

                # Run episode
                obs = env.reset()
                total_reward = 0

                for step in range(300):
                    actions = np.zeros(50, dtype=int)
                    for i in range(50):
                        if not env.agents_reached_goal[i]:
                            actions[i] = policy.get_action(i, obs[i], env)

                    obs, rewards, done, info = env.step(actions)
                    total_reward += np.mean(rewards)

                    if done:
                        break

                episode_results.append(
                    {
                        "steps": step + 1,
                        "success_rate": info["agents_at_goal"] / 50,
                        "total_reward": total_reward,
                        "congestion": np.mean(info["congestion_map"]),
                    }
                )

            # Average results
            avg_results = {
                "success_rate": np.mean([r["success_rate"] for r in episode_results]),
                "steps": np.mean([r["steps"] for r in episode_results]),
                "reward": np.mean([r["total_reward"] for r in episode_results]),
                "congestion": np.mean([r["congestion"] for r in episode_results]),
            }

            results[policy_name] = avg_results

            print(f"   Success Rate: {avg_results['success_rate']:.3f}")
            print(f"   Avg Steps: {avg_results['steps']:.0f}")
            print(f"   Total Reward: {avg_results['reward']:.1f}")
            print(f"   Avg Congestion: {avg_results['congestion']:.3f}")

        # Find best policies
        best_success = max(results.keys(), key=lambda k: results[k]["success_rate"])
        best_efficiency = min(results.keys(), key=lambda k: results[k]["steps"])
        best_reward = max(results.keys(), key=lambda k: results[k]["reward"])

        print("\n🏆 Best Performance:")
        print(f"   Highest Success Rate: {best_success} ({results[best_success]['success_rate']:.3f})")
        print(f"   Most Efficient: {best_efficiency} ({results[best_efficiency]['steps']:.0f} steps)")
        print(f"   Best Reward: {best_reward} ({results[best_reward]['reward']:.1f})")

        return results

    def run_interactive_session(self):
        """Run interactive session with user controls."""
        print("\n🎮 Interactive Page 45 Maze Session")
        print("=" * 40)
        print("Commands:")
        print("  's' - Step forward")
        print("  'r' - Reset environment")
        print("  'p' - Change policy")
        print("  'a' - Auto-run episode")
        print("  'q' - Quit")

        obs = self.env.reset()
        self.env.render()

        while True:
            try:
                command = input("\nEnter command: ").strip().lower()

                if command == "q":
                    break
                elif command == "s":
                    # Single step
                    actions = np.zeros(self.env.config.num_agents, dtype=int)
                    for i in range(self.env.config.num_agents):
                        if not self.env.agents_reached_goal[i]:
                            actions[i] = self.policy.get_action(i, obs[i], self.env)

                    obs, _rewards, done, info = self.env.step(actions)
                    self.env.render()

                    print(
                        f"Step {self.env.current_step}: {info['agents_at_goal']}/{self.env.config.num_agents} at goal"
                    )

                    if done:
                        print("Episode completed!")

                elif command == "r":
                    # Reset
                    obs = self.env.reset()
                    self.env.render()
                    print("Environment reset")

                elif command == "p":
                    # Change policy
                    print("Available policies: random, greedy, congestion, shortest")
                    new_policy = input("Enter policy: ").strip().lower()
                    self.policy = self._create_policy(new_policy)
                    print(f"Switched to {self.policy.name} policy")

                elif command == "a":
                    # Auto-run episode
                    result = self.run_episode(max_steps=500, visualize=True)
                    print(f"Episode result: {result['success_rate']:.3f} success rate")

                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Session ended")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Page 45 Maze Demo")

    parser.add_argument("--agents", type=int, default=50, help="Number of agents")

    parser.add_argument(
        "--policy",
        type=str,
        default="greedy",
        choices=["random", "greedy", "congestion", "shortest"],
        help="Policy to use",
    )

    parser.add_argument("--interactive", action="store_true", help="Run interactive session")

    parser.add_argument("--analysis", action="store_true", help="Run maze analysis")

    parser.add_argument("--comparison", action="store_true", help="Compare all policies")

    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")

    args = parser.parse_args()

    print("📄 Page 45 Paper-Style Maze Demo")
    print("=" * 50)

    # Create demo
    demo = InteractiveMazeDemo(args.agents, args.policy)

    if args.analysis:
        demo.analyze_maze_properties()

    if args.comparison:
        demo.compare_policies()

    if args.interactive:
        demo.run_interactive_session()
    else:
        # Run standard episodes
        for episode in range(args.episodes):
            print(f"\n🎯 Running Episode {episode + 1}/{args.episodes}")
            result = demo.run_episode(max_steps=500, visualize=True)

            print(f"📊 Episode {episode + 1} Results:")
            print(f"   Success Rate: {result['success_rate']:.3f}")
            print(f"   Total Steps: {result['total_steps']}")
            print(f"   Total Reward: {result['total_reward']:.1f}")
            print(f"   Avg Congestion: {result['avg_congestion']:.3f}")

    print("\n✅ Demo Complete!")


if __name__ == "__main__":
    main()
