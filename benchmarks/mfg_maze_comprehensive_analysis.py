#!/usr/bin/env python3
"""
Comprehensive MFG Maze Environment Test Suite

This script provides comprehensive testing of the MFG maze environment including:
- Different maze layouts and configurations
- Various simple policies (random, greedy, shortest path)
- Performance analysis and visualization
- Mean field interaction analysis
- Page 45 paper-style labyrinth testing

Usage:
    python test_mfg_maze_comprehensive.py --layout paper_page45 --agents 100
    python test_mfg_maze_comprehensive.py --layout all --visualization
    python test_mfg_maze_comprehensive.py --policy comparison

Author: MFG_PDE Team
Date: October 2025
"""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Import our maze modules
from mfg_pde.alg.reinforcement.environments.mfg_maze_env import MFGMazeEnvironment
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Import maze analysis tools from examples
# Note: This benchmark script depends on advanced maze examples
try:
    from examples.advanced.mfg_maze_layouts import (
        CustomMazeLoader,
        MazeAnalyzer,
        create_custom_maze_environment,
    )
except ImportError:
    # Fallback: Define minimal stubs if examples not available
    class CustomMazeLoader:
        pass

    class MazeAnalyzer:
        pass

    def create_custom_maze_environment(*args, **kwargs):
        raise NotImplementedError("Maze examples not available")


logger = get_logger(__name__)


class MazePolicy:
    """Base class for maze navigation policies."""

    def __init__(self, name: str):
        self.name = name

    def get_action(self, agent_id: int, observation: np.ndarray, env: MFGMazeEnvironment) -> int:
        """Get action for agent given observation and environment state."""
        raise NotImplementedError


class RandomPolicy(MazePolicy):
    """Random action policy."""

    def __init__(self):
        super().__init__("Random")

    def get_action(self, agent_id: int, observation: np.ndarray, env: MFGMazeEnvironment) -> int:
        return np.random.randint(0, 5)


class GreedyPolicy(MazePolicy):
    """Greedy policy that moves toward goal."""

    def __init__(self):
        super().__init__("Greedy")

    def get_action(self, agent_id: int, observation: np.ndarray, env: MFGMazeEnvironment) -> int:
        agent_pos = env.agent_positions[agent_id]
        goal_pos = env.agent_goals[agent_id]

        # Calculate direction to goal
        diff = goal_pos - agent_pos

        # Choose action based on largest difference
        if abs(diff[0]) > abs(diff[1]):
            # Move vertically
            if diff[0] > 0:
                action = 2  # down
            else:
                action = 1  # up
        else:
            # Move horizontally
            if diff[1] > 0:
                action = 4  # right
            else:
                action = 3  # left

        # Check if action is valid
        action_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        delta = action_map[action]
        new_pos = agent_pos + np.array(delta)

        # Check bounds and walls
        if (
            0 <= new_pos[0] < env.config.height
            and 0 <= new_pos[1] < env.config.width
            and env.maze[new_pos[0], new_pos[1]] != 0
        ):  # Not a wall
            return action
        else:
            # If greedy action is invalid, try other actions
            for try_action in [1, 2, 3, 4, 0]:
                delta = action_map[try_action]
                new_pos = agent_pos + np.array(delta)
                if (
                    0 <= new_pos[0] < env.config.height
                    and 0 <= new_pos[1] < env.config.width
                    and env.maze[new_pos[0], new_pos[1]] != 0
                ):
                    return try_action
            return 0  # Stay in place


class CongestionAwarePolicy(MazePolicy):
    """Policy that avoids congested areas while moving toward goal."""

    def __init__(self, congestion_weight: float = 0.5):
        super().__init__(f"CongestionAware({congestion_weight})")
        self.congestion_weight = congestion_weight

    def get_action(self, agent_id: int, observation: np.ndarray, env: MFGMazeEnvironment) -> int:
        agent_pos = env.agent_positions[agent_id]
        goal_pos = env.agent_goals[agent_id]

        # Evaluate all possible actions
        action_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        best_action = 0
        best_score = float("-inf")

        for action, delta in action_map.items():
            new_pos = agent_pos + np.array(delta)

            # Check if valid move
            if not (
                0 <= new_pos[0] < env.config.height
                and 0 <= new_pos[1] < env.config.width
                and env.maze[new_pos[0], new_pos[1]] != 0
            ):
                continue

            # Calculate score
            # Goal progress component
            old_distance = np.linalg.norm(agent_pos - goal_pos)
            new_distance = np.linalg.norm(new_pos - goal_pos)
            progress_score = old_distance - new_distance

            # Congestion component
            congestion = env._compute_local_congestion(new_pos)
            congestion_score = -congestion * self.congestion_weight

            total_score = progress_score + congestion_score

            if total_score > best_score:
                best_score = total_score
                best_action = action

        return best_action


class ShortestPathPolicy(MazePolicy):
    """Policy that follows shortest path to goal."""

    def __init__(self):
        super().__init__("ShortestPath")
        self.paths = {}  # Cache for computed paths

    def get_action(self, agent_id: int, observation: np.ndarray, env: MFGMazeEnvironment) -> int:
        agent_pos = tuple(env.agent_positions[agent_id])
        goal_pos = tuple(env.agent_goals[agent_id])

        # Check cache
        cache_key = (agent_pos, goal_pos)
        if cache_key not in self.paths:
            # Compute shortest path
            analyzer = MazeAnalyzer(env.maze)
            path_info = analyzer.compute_path_metrics(agent_pos, goal_pos)
            if path_info["path_exists"]:
                self.paths[cache_key] = path_info["shortest_path"]
            else:
                self.paths[cache_key] = None

        path = self.paths[cache_key]
        if path is None or len(path) < 2:
            # No path or already at goal, use greedy policy
            greedy = GreedyPolicy()
            return greedy.get_action(agent_id, observation, env)

        # Find current position in path
        current_idx = None
        for i, pos in enumerate(path):
            if pos == agent_pos:
                current_idx = i
                break

        if current_idx is None or current_idx >= len(path) - 1:
            # Not on path or at end, use greedy
            greedy = GreedyPolicy()
            return greedy.get_action(agent_id, observation, env)

        # Move to next position in path
        next_pos = path[current_idx + 1]
        diff = np.array(next_pos) - np.array(agent_pos)

        # Convert to action
        if np.array_equal(diff, [-1, 0]):
            return 1  # up
        elif np.array_equal(diff, [1, 0]):
            return 2  # down
        elif np.array_equal(diff, [0, -1]):
            return 3  # left
        elif np.array_equal(diff, [0, 1]):
            return 4  # right
        else:
            return 0  # stay


class MazePolicyTester:
    """Comprehensive tester for maze policies."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.results = defaultdict(list)

    def test_policy_on_layout(
        self, policy: MazePolicy, layout_name: str, num_agents: int = 50, max_steps: int = 500, num_episodes: int = 5
    ) -> dict:
        """Test a policy on a specific maze layout."""
        self.logger.info(f"Testing {policy.name} on {layout_name}")

        # Create environment
        env = create_custom_maze_environment(layout_name, num_agents, max_steps)

        episode_results = []

        for _ in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            episode_steps = 0
            congestion_history = []

            for _ in range(max_steps):
                # Get actions from policy for all agents
                actions = np.zeros(num_agents, dtype=int)
                for i in range(num_agents):
                    if not env.agents_reached_goal[i]:
                        actions[i] = policy.get_action(i, obs[i], env)

                # Execute step
                obs, rewards, done, info = env.step(actions)
                total_reward += np.mean(rewards)
                episode_steps += 1

                # Track congestion
                congestion_map = info["congestion_map"]
                avg_congestion = np.mean(congestion_map[congestion_map > 0])
                congestion_history.append(avg_congestion)

                if done:
                    break

            # Episode results
            episode_result = {
                "total_reward": total_reward,
                "steps": episode_steps,
                "agents_reached_goal": info["agents_at_goal"],
                "success_rate": info["agents_at_goal"] / num_agents,
                "mean_distance_to_goal": info["mean_distance_to_goal"],
                "average_congestion": np.mean(congestion_history),
                "max_congestion": np.max(congestion_history) if congestion_history else 0,
            }
            episode_results.append(episode_result)

        # Aggregate results
        aggregated = {
            "policy": policy.name,
            "layout": layout_name,
            "num_episodes": num_episodes,
            "avg_total_reward": np.mean([r["total_reward"] for r in episode_results]),
            "avg_steps": np.mean([r["steps"] for r in episode_results]),
            "avg_success_rate": np.mean([r["success_rate"] for r in episode_results]),
            "avg_final_distance": np.mean([r["mean_distance_to_goal"] for r in episode_results]),
            "avg_congestion": np.mean([r["average_congestion"] for r in episode_results]),
            "episode_results": episode_results,
        }

        return aggregated

    def compare_policies(
        self, policies: list[MazePolicy], layout_name: str, num_agents: int = 50, max_steps: int = 500
    ) -> dict:
        """Compare multiple policies on the same layout."""
        self.logger.info(f"Comparing {len(policies)} policies on {layout_name}")

        comparison_results = {}

        for policy in policies:
            result = self.test_policy_on_layout(policy, layout_name, num_agents, max_steps)
            comparison_results[policy.name] = result

        # Add comparison metrics
        success_rates = {name: result["avg_success_rate"] for name, result in comparison_results.items()}
        rewards = {name: result["avg_total_reward"] for name, result in comparison_results.items()}
        steps = {name: result["avg_steps"] for name, result in comparison_results.items()}

        comparison_results["comparison"] = {
            "best_success_rate": max(success_rates, key=success_rates.get),
            "best_reward": max(rewards, key=rewards.get),
            "fastest": min(steps, key=steps.get),
            "success_rates": success_rates,
            "rewards": rewards,
            "steps": steps,
        }

        return comparison_results

    def test_layout_analysis(self, layout_name: str) -> dict:
        """Analyze layout properties and their impact on policies."""
        loader = CustomMazeLoader()
        maze = loader.get_predefined_layout(layout_name)
        analyzer = MazeAnalyzer(maze)

        # Get maze statistics
        stats = analyzer.get_maze_statistics()

        # Test with different agent densities
        density_results = {}
        agent_counts = [20, 50, 100]

        for num_agents in agent_counts:
            self.logger.info(f"Testing {layout_name} with {num_agents} agents")

            # Test random policy as baseline
            random_policy = RandomPolicy()
            result = self.test_policy_on_layout(random_policy, layout_name, num_agents=num_agents, num_episodes=3)

            density_results[num_agents] = {
                "success_rate": result["avg_success_rate"],
                "congestion": result["avg_congestion"],
                "reward": result["avg_total_reward"],
            }

        return {
            "layout_name": layout_name,
            "maze_statistics": stats,
            "density_analysis": density_results,
        }

    def visualize_policy_comparison(self, comparison_results: dict, save_path: str | None = None):
        """Visualize policy comparison results."""
        policies = [name for name in comparison_results if name != "comparison"]

        # Extract metrics
        success_rates = [comparison_results[policy]["avg_success_rate"] for policy in policies]
        rewards = [comparison_results[policy]["avg_total_reward"] for policy in policies]
        steps = [comparison_results[policy]["avg_steps"] for policy in policies]
        congestion = [comparison_results[policy]["avg_congestion"] for policy in policies]

        # Create subplot
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Success rates
        bars1 = ax1.bar(policies, success_rates, color="skyblue", alpha=0.7)
        ax1.set_title("Success Rate by Policy")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, value in zip(bars1, success_rates, strict=False):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.3f}", ha="center", va="bottom")

        # Total rewards
        bars2 = ax2.bar(policies, rewards, color="lightgreen", alpha=0.7)
        ax2.set_title("Average Total Reward")
        ax2.set_ylabel("Total Reward")
        ax2.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars2, rewards, strict=False):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(rewards) * 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        # Steps to completion
        bars3 = ax3.bar(policies, steps, color="orange", alpha=0.7)
        ax3.set_title("Average Steps to Completion")
        ax3.set_ylabel("Steps")
        ax3.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars3, steps, strict=False):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(steps) * 0.01,
                f"{value:.0f}",
                ha="center",
                va="bottom",
            )

        # Congestion levels
        bars4 = ax4.bar(policies, congestion, color="salmon", alpha=0.7)
        ax4.set_title("Average Congestion Level")
        ax4.set_ylabel("Congestion")
        ax4.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars4, congestion, strict=False):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(congestion) * 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved visualization to {save_path}")

        plt.show()

    def run_comprehensive_test(self, layout_names: list[str] | None = None) -> dict:
        """Run comprehensive test suite across multiple layouts and policies."""
        if layout_names is None:
            loader = CustomMazeLoader()
            layout_names = loader.list_predefined_layouts()

        # Define test policies
        policies = [
            RandomPolicy(),
            GreedyPolicy(),
            CongestionAwarePolicy(0.5),
            CongestionAwarePolicy(1.0),
            ShortestPathPolicy(),
        ]

        comprehensive_results = {}

        for layout_name in layout_names:
            self.logger.info(f"\n🎯 Testing layout: {layout_name}")

            try:
                # Compare policies
                comparison = self.compare_policies(policies, layout_name, num_agents=50)

                # Analyze layout
                layout_analysis = self.test_layout_analysis(layout_name)

                comprehensive_results[layout_name] = {
                    "policy_comparison": comparison,
                    "layout_analysis": layout_analysis,
                }

                # Print summary
                best_policy = comparison["comparison"]["best_success_rate"]
                best_rate = comparison["comparison"]["success_rates"][best_policy]
                self.logger.info(f"   Best policy: {best_policy} ({best_rate:.3f} success rate)")

            except Exception as e:
                self.logger.error(f"   Failed to test {layout_name}: {e}")
                comprehensive_results[layout_name] = {"error": str(e)}

        return comprehensive_results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="MFG Maze Environment Comprehensive Test")

    parser.add_argument(
        "--layout", type=str, default="paper_page45", help="Maze layout to test (or 'all' for all layouts)"
    )

    parser.add_argument("--agents", type=int, default=50, help="Number of agents")

    parser.add_argument("--steps", type=int, default=500, help="Maximum steps per episode")

    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per test")

    parser.add_argument(
        "--policy",
        type=str,
        default="comparison",
        choices=["random", "greedy", "congestion", "shortest", "comparison"],
        help="Policy to test",
    )

    parser.add_argument("--visualization", action="store_true", help="Show visualizations")

    parser.add_argument("--save", type=str, help="Path to save results")

    args = parser.parse_args()

    # Setup logging
    configure_research_logging("mfg_maze_test", level="INFO")

    print("🧪 MFG Maze Environment Comprehensive Test")
    print("=" * 50)

    # Create tester
    tester = MazePolicyTester()

    if args.layout == "all":
        print("🔄 Running comprehensive test on all layouts...")
        results = tester.run_comprehensive_test()

        # Print summary
        print("\n📊 Comprehensive Test Results:")
        for layout_name, layout_results in results.items():
            if "error" not in layout_results:
                comparison = layout_results["policy_comparison"]["comparison"]
                best_policy = comparison["best_success_rate"]
                best_rate = comparison["success_rates"][best_policy]
                print(f"   {layout_name}: {best_policy} ({best_rate:.3f})")
            else:
                print(f"   {layout_name}: ERROR")

    else:
        print(f"🎯 Testing layout: {args.layout}")

        if args.policy == "comparison":
            # Compare all policies
            policies = [
                RandomPolicy(),
                GreedyPolicy(),
                CongestionAwarePolicy(0.5),
                ShortestPathPolicy(),
            ]

            results = tester.compare_policies(policies, args.layout, args.agents, args.steps)

            # Print results
            print("\n📈 Policy Comparison Results:")
            comparison = results["comparison"]
            print(
                f"   Best Success Rate: {comparison['best_success_rate']} "
                f"({comparison['success_rates'][comparison['best_success_rate']]:.3f})"
            )
            print(
                f"   Best Reward: {comparison['best_reward']} "
                f"({comparison['rewards'][comparison['best_reward']]:.1f})"
            )
            print(f"   Fastest: {comparison['fastest']} " f"({comparison['steps'][comparison['fastest']]:.0f} steps)")

            if args.visualization:
                tester.visualize_policy_comparison(results)

        else:
            # Test single policy
            policy_map = {
                "random": RandomPolicy(),
                "greedy": GreedyPolicy(),
                "congestion": CongestionAwarePolicy(1.0),
                "shortest": ShortestPathPolicy(),
            }

            policy = policy_map[args.policy]
            result = tester.test_policy_on_layout(policy, args.layout, args.agents, args.steps, args.episodes)

            print(f"\n📊 {policy.name} Policy Results:")
            print(f"   Success Rate: {result['avg_success_rate']:.3f}")
            print(f"   Average Reward: {result['avg_total_reward']:.1f}")
            print(f"   Average Steps: {result['avg_steps']:.0f}")
            print(f"   Average Congestion: {result['avg_congestion']:.3f}")

    # Test page 45 specific analysis
    if args.layout == "paper_page45" or args.layout == "all":
        print("\n📄 Page 45 Paper Layout Analysis:")

        # Load and analyze the paper layout
        loader = CustomMazeLoader()
        paper_maze = loader.get_predefined_layout("paper_page45")
        analyzer = MazeAnalyzer(paper_maze)

        stats = analyzer.get_maze_statistics()
        print(f"   Dimensions: {stats['dimensions']}")
        print(f"   Wall Density: {stats['wall_density']:.3f}")
        print(f"   Connected Components: {stats['connectivity']['connected_components']}")
        print(f"   Bottlenecks: {len(stats['bottlenecks']['bottlenecks'])}")
        print(f"   Critical Bottlenecks: {len(stats['bottlenecks']['critical_bottlenecks'])}")

        # Test congestion effects with different agent densities
        print("\n🚶 Agent Density Analysis:")
        for agents in [25, 50, 100]:
            random_policy = RandomPolicy()
            result = tester.test_policy_on_layout(random_policy, "paper_page45", agents, 300, 3)
            print(
                f"   {agents} agents: {result['avg_success_rate']:.3f} success, "
                f"{result['avg_congestion']:.3f} congestion"
            )

    if args.save:
        import json

        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.save}")

    print("\n✅ Test Complete!")


if __name__ == "__main__":
    main()
