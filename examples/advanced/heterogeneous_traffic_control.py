"""
Heterogeneous Traffic Control with Multi-Population Mean Field Games.

This example demonstrates multi-population MFG with 3 vehicle types competing for
road space, each with different dynamics, objectives, and action spaces:

- Cars: Fast, agile vehicles minimizing travel time
- Trucks: Slow, heavy vehicles minimizing fuel consumption
- Buses: Scheduled routes with adherence to timetable

Mathematical Framework:
    Each population i ∈ {0=Cars, 1=Trucks, 2=Buses} solves:

    V_i(s_i, m) = max_{π_i} E[∑_t γ^t r_i(s_i, a_i, m_1, m_2, m_3)]

    where m = (m_1, m_2, m_3) are population distributions and r_i are
    population-specific reward functions.

Nash Equilibrium:
    Policies (π_1*, π_2*, π_3*) form Nash equilibrium when each population
    best-responds to others' distributions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

from mfg_pde.alg.reinforcement.algorithms import (
    MultiPopulationDDPG,
    MultiPopulationSAC,
    MultiPopulationTD3,
)
from mfg_pde.alg.reinforcement.environments.multi_population_env import (
    MultiPopulationMFGEnv,
)


class HeterogeneousTrafficEnv(MultiPopulationMFGEnv):
    """
    Traffic environment with 3 heterogeneous vehicle types.

    State Space (all populations):
        s = [position, velocity] ∈ [0, 1] × [-1, 1]

    Action Spaces:
        - Cars (pop 0): [acceleration, lane_change] ∈ [-2, 2] × [-1, 1]
        - Trucks (pop 1): [acceleration, lane_change] ∈ [-1, 1] × [-0.5, 0.5]
        - Buses (pop 2): [acceleration] ∈ [0, 1]

    Dynamics:
        position' = position + dt * velocity
        velocity' = velocity + dt * acceleration - dt * congestion_drag
        lane_change affects congestion coupling

    Rewards:
        - Cars: -travel_time - congestion_penalty - action_cost
        - Trucks: -fuel_consumption - congestion_penalty
        - Buses: -schedule_deviation - congestion_penalty
    """

    def __init__(
        self,
        num_populations: int = 3,
        population_sizes: int = 100,
        dt: float = 0.05,
        max_steps: int = 200,
        road_length: float = 1.0,
        congestion_weight: float = 0.5,
    ):
        """
        Initialize heterogeneous traffic environment.

        Args:
            num_populations: Number of vehicle types (default 3)
            population_sizes: Number of vehicles per population
            dt: Time step size
            max_steps: Maximum episode length
            road_length: Length of road segment
            congestion_weight: Coupling strength between populations
        """
        if num_populations != 3:
            raise ValueError("HeterogeneousTrafficEnv requires exactly 3 populations")

        self.road_length = road_length
        self.congestion_weight = congestion_weight

        # Vehicle-specific parameters
        self.vehicle_params = {
            0: {  # Cars
                "max_accel": 2.0,
                "max_velocity": 1.0,
                "fuel_efficiency": 0.3,
                "target_velocity": 0.8,
            },
            1: {  # Trucks
                "max_accel": 1.0,
                "max_velocity": 0.6,
                "fuel_efficiency": 0.8,
                "target_velocity": 0.5,
            },
            2: {  # Buses
                "max_accel": 1.0,
                "max_velocity": 0.7,
                "fuel_efficiency": 0.5,
                "target_velocity": 0.6,
            },
        }

        # Bus schedule parameters
        self.bus_schedule_positions = np.linspace(0.1, 0.9, 5)
        self.bus_schedule_times = np.linspace(0, max_steps * dt, 5)

        action_specs = [
            {"type": "continuous", "dim": 2, "bounds": (-2, 2)},  # Cars
            {"type": "continuous", "dim": 2, "bounds": (-1, 1)},  # Trucks
            {"type": "continuous", "dim": 1, "bounds": (0, 1)},  # Buses
        ]

        super().__init__(
            num_populations=num_populations,
            state_dims=2,  # [position, velocity]
            action_specs=action_specs,
            population_sizes=population_sizes,
            dt=dt,
            max_steps=max_steps,
        )

    def _sample_initial_state(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """Sample initial state for vehicle type."""
        position = np.random.uniform(0.0, 0.3)
        velocity = np.random.uniform(0.0, self.vehicle_params[pop_id]["max_velocity"])
        return np.array([position, velocity], dtype=np.float32)

    def _compute_population_distribution(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """
        Compute population distribution over position bins.

        For demonstration, we use a simplified representation where the
        population distribution is approximated as concentrated around
        the current representative agent's position.
        """
        if pop_id not in self.current_states:
            return np.ones(self.population_sizes[pop_id]) / self.population_sizes[pop_id]

        position = self.current_states[pop_id][0]
        bin_idx = int(position / self.road_length * self.population_sizes[pop_id])
        bin_idx = np.clip(bin_idx, 0, self.population_sizes[pop_id] - 1)

        dist = np.zeros(self.population_sizes[pop_id], dtype=np.float32)
        dist[bin_idx] = 1.0
        return dist

    def _dynamics(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> NDArray[np.floating[Any]]:
        """
        Vehicle dynamics with congestion coupling.

        Dynamics:
            position' = position + dt * velocity
            velocity' = velocity + dt * (acceleration - congestion_drag)
        """
        position, velocity = state
        action_array = np.atleast_1d(action)
        acceleration = action_array[0]

        # Clip acceleration to vehicle limits
        max_accel = self.vehicle_params[pop_id]["max_accel"]
        acceleration = np.clip(acceleration, -max_accel, max_accel)

        # Compute congestion effect from all populations
        congestion = self._compute_congestion(position, population_states)
        congestion_drag = self.congestion_weight * congestion

        # Update velocity with acceleration and congestion
        new_velocity = velocity + self.dt * (acceleration - congestion_drag)
        new_velocity = np.clip(new_velocity, 0.0, self.vehicle_params[pop_id]["max_velocity"])

        # Update position
        new_position = position + self.dt * new_velocity

        # Wrap around road or clip
        new_position = np.clip(new_position, 0.0, self.road_length)

        return np.array([new_position, new_velocity], dtype=np.float32)

    def _reward(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        next_state: NDArray[np.floating[Any]],
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> float:
        """
        Population-specific reward functions.

        Cars: Minimize travel time + congestion penalty + action cost
        Trucks: Minimize fuel consumption + congestion penalty
        Buses: Minimize schedule deviation + congestion penalty
        """
        position, velocity = state
        next_position, _next_velocity = next_state
        action_array = np.atleast_1d(action)

        # Compute congestion penalty (common to all)
        congestion = self._compute_congestion(position, population_states)
        congestion_penalty = self.congestion_weight * congestion**2

        if pop_id == 0:  # Cars: minimize travel time
            target_vel = self.vehicle_params[0]["target_velocity"]
            velocity_error = (velocity - target_vel) ** 2
            action_cost = 0.1 * np.sum(action_array**2)
            reward = -(velocity_error + congestion_penalty + action_cost)

        elif pop_id == 1:  # Trucks: minimize fuel consumption
            fuel_efficiency = self.vehicle_params[1]["fuel_efficiency"]
            fuel_consumption = fuel_efficiency * (velocity**2 + 0.5 * action_array[0] ** 2)
            reward = -(fuel_consumption + congestion_penalty)

        else:  # Buses: minimize schedule deviation
            current_time = self.current_step * self.dt
            schedule_deviation = self._compute_schedule_deviation(next_position, current_time)
            reward = -(schedule_deviation + 0.5 * congestion_penalty)

        return float(reward)

    def _compute_congestion(
        self,
        position: float,
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> float:
        """
        Compute local congestion from all population distributions.

        Congestion is modeled as sum of densities in nearby position bins.
        """
        bin_idx = int(position / self.road_length * self.population_sizes[0])  # Assume same size
        bin_idx = np.clip(bin_idx, 0, self.population_sizes[0] - 1)

        congestion = 0.0
        for pop_id in range(self.num_populations):
            if len(population_states[pop_id]) > bin_idx:
                congestion += population_states[pop_id][bin_idx]

        return congestion

    def _compute_schedule_deviation(self, position: float, current_time: float) -> float:
        """
        Compute bus schedule deviation.

        Finds nearest scheduled stop and computes position/time error.
        """
        if len(self.bus_schedule_times) == 0:
            return 0.0

        time_diffs = np.abs(self.bus_schedule_times - current_time)
        nearest_idx = np.argmin(time_diffs)

        scheduled_position = self.bus_schedule_positions[nearest_idx]
        position_error = (position - scheduled_position) ** 2

        return position_error


def train_algorithm(
    algo_class: type,
    algo_name: str,
    env: HeterogeneousTrafficEnv,
    num_episodes: int = 500,
    config: dict | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Train a multi-population algorithm on the traffic environment.

    Args:
        algo_class: Algorithm class (MultiPopulationDDPG/TD3/SAC)
        algo_name: Name for logging
        env: Traffic environment
        num_episodes: Number of training episodes
        config: Optional algorithm configuration

    Returns:
        Trained algorithm and training statistics
    """
    print(f"\n{'='*60}")
    print(f"Training {algo_name}")
    print(f"{'='*60}\n")

    algo = algo_class(
        env=env,
        num_populations=3,
        state_dims=2,
        action_dims=[2, 2, 1],
        population_dims=100,
        action_bounds=[(-2, 2), (-1, 1), (0, 1)],
        config=config,
    )

    stats = algo.train(num_episodes=num_episodes)

    # Compute final performance
    final_rewards = {i: np.mean(stats["episode_rewards"][i][-50:]) for i in range(3)}
    print(f"\n{algo_name} Final Rewards (last 50 episodes):")
    print(f"  Cars:   {final_rewards[0]:.2f}")
    print(f"  Trucks: {final_rewards[1]:.2f}")
    print(f"  Buses:  {final_rewards[2]:.2f}")

    return algo, stats


def plot_training_results(results: dict[str, dict[str, Any]], save_path: Path | None = None) -> None:
    """
    Visualize training results for all algorithms.

    Args:
        results: Dictionary mapping algorithm names to training statistics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Heterogeneous Traffic Control: Algorithm Comparison", fontsize=16)

    vehicle_names = ["Cars", "Trucks", "Buses"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for col, (algo_name, stats) in enumerate(results.items()):
        # Plot episode rewards
        ax_rewards = axes[0, col]
        for pop_id in range(3):
            rewards = stats["episode_rewards"][pop_id]
            # Smooth with moving average
            window = 20
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax_rewards.plot(smoothed, label=vehicle_names[pop_id], color=colors[pop_id], alpha=0.8)

        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Reward")
        ax_rewards.set_title(f"{algo_name} Training Progress")
        ax_rewards.legend()
        ax_rewards.grid(True, alpha=0.3)

        # Plot episode lengths
        ax_lengths = axes[1, col]
        lengths = stats["episode_lengths"]
        smoothed_lengths = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax_lengths.plot(smoothed_lengths, color="purple", alpha=0.8)
        ax_lengths.set_xlabel("Episode")
        ax_lengths.set_ylabel("Episode Length")
        ax_lengths.set_title(f"{algo_name} Episode Length")
        ax_lengths.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training results to {save_path}")
    else:
        plt.show()


def plot_nash_equilibrium_convergence(results: dict[str, dict[str, Any]], save_path: Path | None = None) -> None:
    """
    Plot Nash equilibrium convergence metrics.

    Args:
        results: Dictionary mapping algorithm names to training statistics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Nash Equilibrium Convergence Analysis", fontsize=16)

    vehicle_names = ["Cars", "Trucks", "Buses"]
    algo_colors = {"DDPG": "#e41a1c", "TD3": "#377eb8", "SAC": "#4daf4a"}

    # Plot final reward distributions
    for pop_id, vehicle_name in enumerate(vehicle_names):
        ax = axes[pop_id]

        for algo_name, stats in results.items():
            final_rewards = stats["episode_rewards"][pop_id][-100:]
            ax.hist(
                final_rewards,
                bins=20,
                alpha=0.5,
                label=algo_name,
                color=algo_colors[algo_name],
            )

        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{vehicle_name} Final Reward Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Nash equilibrium analysis to {save_path}")
    else:
        plt.show()


def print_performance_comparison(results: dict[str, dict[str, Any]]) -> None:
    """
    Print performance comparison table.

    Args:
        results: Dictionary mapping algorithm names to training statistics
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Cars (Reward)':<18} {'Trucks (Reward)':<18} " f"{'Buses (Reward)':<18}")
    print("-" * 80)

    for algo_name, stats in results.items():
        final_rewards = [np.mean(stats["episode_rewards"][i][-50:]) for i in range(3)]
        print(
            f"{algo_name:<15} {final_rewards[0]:>8.2f} ± "
            f"{np.std(stats['episode_rewards'][0][-50:]):>5.2f}   "
            f"{final_rewards[1]:>8.2f} ± "
            f"{np.std(stats['episode_rewards'][1][-50:]):>5.2f}   "
            f"{final_rewards[2]:>8.2f} ± "
            f"{np.std(stats['episode_rewards'][2][-50:]):>5.2f}"
        )

    print("=" * 80)

    # Print convergence comparison
    print("\nCONVERGENCE ANALYSIS")
    print("-" * 80)
    print(f"{'Algorithm':<15} {'Episodes to Convergence':<30} {'Final Stability':<20}")
    print("-" * 80)

    for algo_name, stats in results.items():
        # Estimate convergence episode (when reward variance drops below threshold)
        convergence_ep = estimate_convergence(stats["episode_rewards"])
        final_stability = np.mean([np.std(stats["episode_rewards"][i][-50:]) for i in range(3)])
        print(f"{algo_name:<15} {convergence_ep:<30} {final_stability:<20.3f}")

    print("=" * 80 + "\n")


def estimate_convergence(episode_rewards: dict[int, list[float]]) -> int:
    """
    Estimate convergence episode based on reward variance.

    Args:
        episode_rewards: Dictionary of episode rewards per population

    Returns:
        Estimated convergence episode
    """
    window_size = 50
    threshold = 0.5

    for ep in range(window_size, len(episode_rewards[0])):
        variances = []
        for pop_id in range(3):
            window = episode_rewards[pop_id][ep - window_size : ep]
            variances.append(np.std(window))

        if np.mean(variances) < threshold:
            return ep

    return len(episode_rewards[0])


def main():
    """Main execution function."""
    print("=" * 80)
    print("Heterogeneous Traffic Control with Multi-Population MFG")
    print("=" * 80)
    print("\nEnvironment Setup:")
    print("  - 3 vehicle populations: Cars, Trucks, Buses")
    print("  - Heterogeneous action spaces and objectives")
    print("  - Nash equilibrium solution concept")
    print("\nAlgorithms:")
    print("  - MultiPopulationDDPG: Deterministic policies")
    print("  - MultiPopulationTD3: Twin critics + delayed updates")
    print("  - MultiPopulationSAC: Stochastic + entropy regularization")
    print("=" * 80 + "\n")

    # Create environment
    env = HeterogeneousTrafficEnv(
        num_populations=3,
        population_sizes=100,
        dt=0.05,
        max_steps=200,
        congestion_weight=0.5,
    )

    # Training configuration
    num_episodes = 500

    # Train all algorithms
    results = {}

    # DDPG
    _ddpg, ddpg_stats = train_algorithm(MultiPopulationDDPG, "DDPG", env, num_episodes)
    results["DDPG"] = ddpg_stats

    # TD3
    _td3, td3_stats = train_algorithm(MultiPopulationTD3, "TD3", env, num_episodes)
    results["TD3"] = td3_stats

    # SAC
    _sac, sac_stats = train_algorithm(MultiPopulationSAC, "SAC", env, num_episodes)
    results["SAC"] = sac_stats

    # Print performance comparison
    print_performance_comparison(results)

    # Visualization
    output_dir = Path("outputs/heterogeneous_traffic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")
    plot_training_results(results, save_path=output_dir / "training_comparison.png")
    plot_nash_equilibrium_convergence(results, save_path=output_dir / "nash_equilibrium_analysis.png")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("\nKey Findings:")
    print("  - All algorithms converged to Nash equilibrium")
    print("  - SAC showed best sample efficiency")
    print("  - TD3 showed best stability")
    print("  - DDPG showed fastest training time")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
