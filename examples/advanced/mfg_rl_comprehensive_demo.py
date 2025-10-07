#!/usr/bin/env python3
"""
Comprehensive MFG RL Experiment Demo

This script demonstrates a complete experimental setup for Mean Field Games
Reinforcement Learning, including multiple scenarios and algorithm comparisons.

Features:
- Multiple MFG scenarios (crowd navigation, linear-quadratic, etc.)
- Mean Field Q-Learning implementation
- Baseline comparisons
- Performance evaluation and visualization
- Extensible framework for new algorithms

Usage:
    python mfg_rl_comprehensive_demo.py --scenario crowd_navigation --episodes 2000
    python mfg_rl_comprehensive_demo.py --scenario linear_quadratic --episodes 1500
    python mfg_rl_comprehensive_demo.py --scenario finite_state --episodes 1000

Author: MFG_PDE Team
Date: October 2025
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# MFG_PDE imports
from examples.advanced.mfg_rl_experiment_suite import ExperimentConfig, MFGRLExperimentSuite
from mfg_pde.alg.reinforcement.algorithms.mean_field_q_learning import create_mean_field_q_learning
from mfg_pde.utils.logging import configure_research_logging, get_logger

logger = get_logger(__name__)


class MFGRLDemoRunner:
    """Comprehensive demo runner for MFG RL experiments."""

    def __init__(self, args):
        self.args = args
        self.logger = get_logger(__name__)

        # Setup logging
        configure_research_logging(f"mfg_rl_demo_{args.scenario}", level="INFO")

        # Create experiment configuration
        self.config = ExperimentConfig(
            scenario=args.scenario,
            num_agents=args.num_agents,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            domain_size=args.domain_size,
            grid_resolution=args.grid_resolution,
            learning_rate=args.learning_rate,
            save_results=True,
            plot_results=True,
        )

        # Initialize experiment suite
        self.suite = MFGRLExperimentSuite(self.config)

        # Results storage
        self.results = {}

    def run_full_experiment(self) -> dict:
        """Run complete experimental evaluation."""
        self.logger.info("🚀 Starting Comprehensive MFG RL Experiment")
        self.logger.info(f"📊 Scenario: {self.config.scenario}")
        self.logger.info(f"👥 Agents: {self.config.num_agents}")
        self.logger.info(f"🎯 Episodes: {self.config.num_episodes}")

        start_time = time.time()

        # 1. Run random baseline
        self.logger.info("\n📈 Phase 1: Random Baseline Evaluation")
        self.results["random_baseline"] = self.suite.run_random_baseline()
        self._log_results("Random Baseline", self.results["random_baseline"])

        # 2. Run Mean Field Q-Learning
        self.logger.info("\n🧠 Phase 2: Mean Field Q-Learning")
        mfql_config = {
            "learning_rate": self.config.learning_rate,
            "epsilon_decay": 0.995,
            "batch_size": 64,
            "target_update_frequency": 100,
        }

        try:
            mfql_algorithm = create_mean_field_q_learning(self.suite.env, mfql_config)
            self.results["mean_field_q_learning"] = self.suite.evaluate_mfrl_algorithm(
                type(mfql_algorithm), {"algorithm": mfql_algorithm}
            )
            self._log_results("Mean Field Q-Learning", self.results["mean_field_q_learning"])

        except Exception as e:
            self.logger.error(f"Mean Field Q-Learning failed: {e}")
            self.results["mean_field_q_learning"] = {"error": str(e)}

        # 3. Generate comparison report
        self.logger.info("\n📋 Phase 3: Comparative Analysis")
        comparison_report = self._generate_comprehensive_report()
        self.results["comprehensive_analysis"] = comparison_report

        # 4. Save results
        self._save_results()

        # 5. Generate visualizations
        self._generate_visualizations()

        total_time = time.time() - start_time
        self.logger.info(f"\n✅ Experiment Complete! Total time: {total_time:.2f}s")

        return self.results

    def run_scenario_comparison(self) -> dict:
        """Run comparison across multiple scenarios."""
        scenarios = ["crowd_navigation", "linear_quadratic", "finite_state"]
        scenario_results = {}

        self.logger.info("🔄 Running Multi-Scenario Comparison")

        for scenario in scenarios:
            self.logger.info(f"\n🎯 Testing Scenario: {scenario}")

            # Update config for this scenario
            scenario_config = ExperimentConfig(
                scenario=scenario,
                num_agents=min(self.config.num_agents, 50),  # Smaller for comparison
                num_episodes=min(self.config.num_episodes, 500),  # Faster for comparison
                max_steps_per_episode=100,
                save_results=False,
                plot_results=False,
            )

            # Create suite for this scenario
            scenario_suite = MFGRLExperimentSuite(scenario_config)

            try:
                # Run baseline
                baseline = scenario_suite.run_random_baseline()

                # Run MFQL
                mfql_algorithm = create_mean_field_q_learning(scenario_suite.env)
                mfql_results = scenario_suite.evaluate_mfrl_algorithm(
                    type(mfql_algorithm), {"algorithm": mfql_algorithm}
                )

                scenario_results[scenario] = {
                    "baseline": baseline,
                    "mfql": mfql_results,
                    "improvement": self._calculate_improvement(baseline, mfql_results),
                }

                self.logger.info(f"✅ {scenario} complete")

            except Exception as e:
                self.logger.error(f"❌ {scenario} failed: {e}")
                scenario_results[scenario] = {"error": str(e)}

        return scenario_results

    def _log_results(self, algorithm_name: str, results: dict):
        """Log algorithm results."""
        if "error" in results:
            self.logger.error(f"❌ {algorithm_name}: {results['error']}")
            return

        if algorithm_name == "Random Baseline":
            final_reward = results.get("final_average_reward", 0)
            convergence = results.get("convergence_achieved", False)
            self.logger.info(f"   Final Reward: {final_reward:.3f}")
            self.logger.info(f"   Convergence: {convergence}")

        else:
            training_results = results.get("training_results", {})
            eval_results = results.get("evaluation_results", {})

            if training_results and "episode_rewards" in training_results:
                final_training_reward = training_results["episode_rewards"][-1]
                self.logger.info(f"   Final Training Reward: {final_training_reward:.3f}")

            if eval_results:
                avg_reward = eval_results.get("average_reward", 0)
                nash_error = eval_results.get("average_nash_error", 0)
                self.logger.info(f"   Evaluation Reward: {avg_reward:.3f}")
                self.logger.info(f"   Nash Error: {nash_error:.6f}")

    def _generate_comprehensive_report(self) -> dict:
        """Generate comprehensive analysis report."""
        report = {
            "scenario_analysis": {},
            "algorithm_comparison": {},
            "convergence_analysis": {},
            "performance_metrics": {},
        }

        # Scenario analysis
        report["scenario_analysis"] = {
            "scenario": self.config.scenario,
            "num_agents": self.config.num_agents,
            "num_episodes": self.config.num_episodes,
            "domain_characteristics": self._analyze_scenario_characteristics(),
        }

        # Algorithm comparison
        algorithms = ["random_baseline", "mean_field_q_learning"]
        performance_data = {}

        for alg in algorithms:
            if alg in self.results and "error" not in self.results[alg]:
                if alg == "random_baseline":
                    performance_data[alg] = self.results[alg].get("final_average_reward", 0)
                else:
                    eval_results = self.results[alg].get("evaluation_results", {})
                    performance_data[alg] = eval_results.get("average_reward", 0)

        report["algorithm_comparison"] = {
            "performance_ranking": sorted(performance_data.items(), key=lambda x: x[1], reverse=True),
            "performance_data": performance_data,
        }

        # Convergence analysis
        if "mean_field_q_learning" in self.results and "training_results" in self.results["mean_field_q_learning"]:
            training_data = self.results["mean_field_q_learning"]["training_results"]
            if "episode_rewards" in training_data:
                report["convergence_analysis"] = self._analyze_convergence(training_data["episode_rewards"])

        # Performance metrics
        report["performance_metrics"] = self._compute_performance_metrics()

        return report

    def _analyze_scenario_characteristics(self) -> dict:
        """Analyze characteristics of the current scenario."""
        characteristics = {"scenario": self.config.scenario}

        if self.config.scenario == "crowd_navigation":
            characteristics.update(
                {
                    "type": "Spatial Navigation",
                    "state_space": "Continuous 2D",
                    "action_space": "Continuous velocities",
                    "interaction_type": "Congestion-based",
                    "difficulty": "Medium",
                }
            )

        elif self.config.scenario == "linear_quadratic":
            characteristics.update(
                {
                    "type": "Control Theory",
                    "state_space": "Continuous multi-dimensional",
                    "action_space": "Continuous control",
                    "interaction_type": "Mean field coupling",
                    "difficulty": "Low (has analytical solution)",
                }
            )

        elif self.config.scenario == "finite_state":
            characteristics.update(
                {
                    "type": "Discrete State Game",
                    "state_space": "Finite discrete",
                    "action_space": "Discrete transitions",
                    "interaction_type": "State occupation coupling",
                    "difficulty": "Medium",
                }
            )

        elif self.config.scenario == "epidemic":
            characteristics.update(
                {
                    "type": "Epidemic Control",
                    "state_space": "Health states + spatial",
                    "action_space": "Isolation decisions",
                    "interaction_type": "Infection network",
                    "difficulty": "High",
                }
            )

        elif self.config.scenario == "price_formation":
            characteristics.update(
                {
                    "type": "Financial Market",
                    "state_space": "Portfolio + market state",
                    "action_space": "Trading decisions",
                    "interaction_type": "Price impact",
                    "difficulty": "High",
                }
            )

        return characteristics

    def _analyze_convergence(self, episode_rewards: list[float]) -> dict:
        """Analyze convergence properties of training."""
        rewards = np.array(episode_rewards)

        # Compute moving averages
        window_sizes = [10, 50, 100]
        moving_averages = {}

        for window in window_sizes:
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
                moving_averages[f"ma_{window}"] = moving_avg.tolist()

        # Convergence detection
        convergence_detected = False
        convergence_episode = None

        if len(rewards) >= 100:
            # Check if last 100 episodes show stability
            recent_rewards = rewards[-100:]
            if np.std(recent_rewards) < 0.1 * np.abs(np.mean(recent_rewards)):
                convergence_detected = True
                # Find approximate convergence point
                for i in range(100, len(rewards)):
                    window_rewards = rewards[i - 100 : i]
                    if np.std(window_rewards) < 0.1 * np.abs(np.mean(window_rewards)):
                        convergence_episode = i
                        break

        return {
            "convergence_detected": convergence_detected,
            "convergence_episode": convergence_episode,
            "moving_averages": moving_averages,
            "final_performance": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(rewards[-1]),
            "performance_trend": self._compute_trend(rewards),
        }

    def _compute_trend(self, rewards: np.ndarray) -> str:
        """Compute performance trend."""
        if len(rewards) < 100:
            return "insufficient_data"

        early_avg = np.mean(rewards[:50])
        late_avg = np.mean(rewards[-50:])

        improvement = (late_avg - early_avg) / (abs(early_avg) + 1e-6)

        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "degrading"
        else:
            return "stable"

    def _compute_performance_metrics(self) -> dict:
        """Compute comprehensive performance metrics."""
        metrics = {}

        # Compare baseline vs MFQL
        if "random_baseline" in self.results and "mean_field_q_learning" in self.results:
            baseline_perf = self.results["random_baseline"].get("final_average_reward", 0)

            mfql_results = self.results["mean_field_q_learning"]
            if "evaluation_results" in mfql_results:
                mfql_perf = mfql_results["evaluation_results"].get("average_reward", 0)

                improvement = (mfql_perf - baseline_perf) / (abs(baseline_perf) + 1e-6)
                metrics["performance_improvement"] = {
                    "absolute": mfql_perf - baseline_perf,
                    "relative": improvement,
                    "baseline_performance": baseline_perf,
                    "mfql_performance": mfql_perf,
                }

        return metrics

    def _calculate_improvement(self, baseline: dict, algorithm: dict) -> dict:
        """Calculate improvement metrics for scenario comparison."""
        baseline_reward = baseline.get("final_average_reward", 0)

        if "evaluation_results" in algorithm:
            alg_reward = algorithm["evaluation_results"].get("average_reward", 0)
        else:
            alg_reward = 0

        improvement = (alg_reward - baseline_reward) / (abs(baseline_reward) + 1e-6)

        return {
            "absolute_improvement": alg_reward - baseline_reward,
            "relative_improvement": improvement,
            "baseline_reward": baseline_reward,
            "algorithm_reward": alg_reward,
        }

    def _save_results(self):
        """Save experimental results."""
        results_dir = Path("mfg_rl_results")
        results_dir.mkdir(exist_ok=True)

        # Save main results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"mfg_rl_results_{self.config.scenario}_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(self.results)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        self.logger.info(f"💾 Results saved to {results_file}")

        # Save configuration
        config_file = results_dir / f"mfg_rl_config_{self.config.scenario}_{timestamp}.json"
        config_dict = {
            "scenario": self.config.scenario,
            "num_agents": self.config.num_agents,
            "num_episodes": self.config.num_episodes,
            "max_steps_per_episode": self.config.max_steps_per_episode,
            "learning_rate": self.config.learning_rate,
            "domain_size": self.config.domain_size,
            "grid_resolution": self.config.grid_resolution,
        }

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32 | np.float64):
            return float(obj)
        elif isinstance(obj, np.int32 | np.int64):
            return int(obj)
        else:
            return obj

    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        if not self.config.plot_results:
            return

        # Main performance comparison plot
        self._plot_performance_comparison()

        # Training curves (if available)
        self._plot_training_curves()

        # Scenario analysis plot
        self._plot_scenario_analysis()

        self.logger.info("📊 Visualizations generated and saved")

    def _plot_performance_comparison(self):
        """Plot performance comparison between algorithms."""
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Performance bar chart
        algorithms = []
        performances = []

        if "random_baseline" in self.results:
            algorithms.append("Random\nBaseline")
            performances.append(self.results["random_baseline"].get("final_average_reward", 0))

        if "mean_field_q_learning" in self.results and "evaluation_results" in self.results["mean_field_q_learning"]:
            algorithms.append("Mean Field\nQ-Learning")
            eval_results = self.results["mean_field_q_learning"]["evaluation_results"]
            performances.append(eval_results.get("average_reward", 0))

        if algorithms:
            bars = ax1.bar(algorithms, performances, color=["red", "blue"][: len(algorithms)], alpha=0.7)
            ax1.set_title(f"Algorithm Performance - {self.config.scenario}")
            ax1.set_ylabel("Average Episode Reward")
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, performances, strict=False):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom")

        # Performance improvement (if applicable)
        if len(performances) >= 2:
            improvement = (performances[1] - performances[0]) / (abs(performances[0]) + 1e-6) * 100
            ax2.bar(["Performance\nImprovement"], [improvement], color="green" if improvement > 0 else "red", alpha=0.7)
            ax2.set_title("Relative Performance Improvement")
            ax2.set_ylabel("Improvement (%)")
            ax2.grid(True, alpha=0.3)
            ax2.text(0, improvement, f"{improvement:.1f}%", ha="center", va="bottom" if improvement > 0 else "top")

        plt.suptitle(f"MFG RL Performance Analysis - {self.config.scenario}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"mfg_rl_performance_{self.config.scenario}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_training_curves(self):
        """Plot training curves if available."""
        if "mean_field_q_learning" not in self.results:
            return

        training_results = self.results["mean_field_q_learning"].get("training_results", {})
        if "episode_rewards" not in training_results:
            return

        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        rewards = training_results["episode_rewards"]
        episodes = range(len(rewards))

        # Episode rewards
        ax1.plot(episodes, rewards, alpha=0.6, color="blue")
        if len(rewards) >= 50:
            moving_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
            ax1.plot(range(49, len(rewards)), moving_avg, color="red", linewidth=2, label="50-episode MA")
            ax1.legend()
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)

        # Loss history (if available)
        if "loss_history" in training_results:
            losses = training_results["loss_history"]
            ax2.plot(losses, color="orange", alpha=0.7)
            ax2.set_title("Training Loss")
            ax2.set_xlabel("Training Step")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

        # Nash errors (if available)
        if "nash_errors" in training_results:
            nash_errors = training_results["nash_errors"]
            ax3.plot(nash_errors, color="green", alpha=0.7)
            ax3.set_title("Nash Equilibrium Error")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Nash Error")
            ax3.grid(True, alpha=0.3)

        # Performance distribution
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            ax4.hist(recent_rewards, bins=20, alpha=0.7, color="purple")
            ax4.set_title("Recent Performance Distribution (Last 100 Episodes)")
            ax4.set_xlabel("Episode Reward")
            ax4.set_ylabel("Frequency")
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f"MFG RL Training Analysis - {self.config.scenario}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"mfg_rl_training_{self.config.scenario}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_scenario_analysis(self):
        """Plot scenario-specific analysis."""
        # This could be extended with scenario-specific visualizations
        # For now, just show basic environment characteristics

        _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Create a simple summary plot
        characteristics = self._analyze_scenario_characteristics()

        info_text = f"Scenario: {characteristics['scenario']}\n"
        info_text += f"Type: {characteristics.get('type', 'Unknown')}\n"
        info_text += f"State Space: {characteristics.get('state_space', 'Unknown')}\n"
        info_text += f"Action Space: {characteristics.get('action_space', 'Unknown')}\n"
        info_text += f"Interaction: {characteristics.get('interaction_type', 'Unknown')}\n"
        info_text += f"Difficulty: {characteristics.get('difficulty', 'Unknown')}\n\n"

        info_text += "Experiment Configuration:\n"
        info_text += f"Agents: {self.config.num_agents}\n"
        info_text += f"Episodes: {self.config.num_episodes}\n"
        info_text += f"Domain Size: {self.config.domain_size}\n"

        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12, verticalalignment="center")
        ax.set_title(f"Scenario Analysis - {self.config.scenario}")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"mfg_rl_scenario_{self.config.scenario}.png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="MFG RL Comprehensive Demo")

    parser.add_argument(
        "--scenario",
        type=str,
        default="crowd_navigation",
        choices=["crowd_navigation", "linear_quadratic", "finite_state", "epidemic", "price_formation"],
        help="MFG scenario to test",
    )

    parser.add_argument("--num_agents", type=int, default=100, help="Number of agents")

    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")

    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")

    parser.add_argument("--domain_size", type=float, default=10.0, help="Domain size")

    parser.add_argument("--grid_resolution", type=int, default=50, help="Grid resolution")

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")

    parser.add_argument("--multi_scenario", action="store_true", help="Run comparison across multiple scenarios")

    args = parser.parse_args()

    print("🎮 MFG RL Comprehensive Demo")
    print("=" * 50)

    # Create demo runner
    demo = MFGRLDemoRunner(args)

    if args.multi_scenario:
        print("🔄 Running Multi-Scenario Comparison...")
        scenario_results = demo.run_scenario_comparison()

        print("\n📊 Multi-Scenario Results:")
        for scenario, results in scenario_results.items():
            if "error" not in results:
                improvement = results["improvement"]["relative_improvement"]
                print(f"   {scenario}: {improvement:+.1%} improvement")
            else:
                print(f"   {scenario}: Failed ({results['error']})")

    else:
        print(f"🎯 Running Single Scenario: {args.scenario}")
        results = demo.run_full_experiment()

        print("\n🎉 Demo Complete!")
        print("\n📋 Quick Summary:")

        if "comprehensive_analysis" in results:
            analysis = results["comprehensive_analysis"]

            if "algorithm_comparison" in analysis and "performance_ranking" in analysis["algorithm_comparison"]:
                ranking = analysis["algorithm_comparison"]["performance_ranking"]
                print("   Performance Ranking:")
                for i, (alg, score) in enumerate(ranking):
                    print(f"      {i + 1}. {alg}: {score:.3f}")

            if "performance_metrics" in analysis and "performance_improvement" in analysis["performance_metrics"]:
                improvement = analysis["performance_metrics"]["performance_improvement"]
                rel_improvement = improvement["relative_improvement"]
                print(f"   MFQL Improvement: {rel_improvement:+.1%}")

        print("\n💡 Next Steps:")
        print("   1. Experiment with different hyperparameters")
        print("   2. Implement additional MFRL algorithms")
        print("   3. Test on custom scenarios")
        print("   4. Scale up to larger agent populations")


if __name__ == "__main__":
    main()
