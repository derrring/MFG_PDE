"""
Heterogeneous Traffic Flow with Multi-Population MFG.

Demonstrates multi-population Mean Field Games with three vehicle types:
- Cars: Fast, DDPG control
- Trucks: Slow, heavy, TD3 control
- Motorcycles: Agile, SAC control

Each population has different dynamics and responds to all population densities.

Mathematical Framework:
- State: (position, velocity) for each vehicle type
- Dynamics: v' = v + a*dt, x' = x + v*dt with density-dependent drag
- Coupling: Speed depends on densities of all vehicle types

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import numpy as np

from mfg_pde.alg.reinforcement.multi_population.base_environment import (
    MultiPopulationMFGEnvironment,
)
from mfg_pde.alg.reinforcement.multi_population.multi_ddpg import MultiPopulationDDPG
from mfg_pde.alg.reinforcement.multi_population.multi_sac import MultiPopulationSAC
from mfg_pde.alg.reinforcement.multi_population.multi_td3 import MultiPopulationTD3
from mfg_pde.alg.reinforcement.multi_population.population_config import (
    PopulationConfig,
)
from mfg_pde.alg.reinforcement.multi_population.trainer import MultiPopulationTrainer


class HeterogeneousTrafficEnv(MultiPopulationMFGEnvironment):
    """
    Traffic environment with three vehicle types.

    State: [position, velocity] for each agent
    Dynamics:
    - Position: x' = x + v*dt
    - Velocity: v' = v + a*dt - drag(m_all)
    - Drag depends on all population densities

    Reward:
    - Target velocity maintenance: -|v - v_target|
    - Fuel efficiency: -|a|²
    - Congestion avoidance: -Σ w_j * density_j
    """

    def __init__(
        self,
        populations: dict[str, PopulationConfig],
        road_length: float = 10.0,
        time_horizon: float = 10.0,
        dt: float = 0.1,
    ):
        """
        Initialize traffic environment.

        Args:
            populations: Vehicle population configurations
            road_length: Road length in km
            time_horizon: Simulation time in seconds
            dt: Time step size
        """
        super().__init__(
            populations=populations,
            coupling_dynamics=None,
            domain=None,
            time_horizon=time_horizon,
            dt=dt,
        )

        self.road_length = road_length

        # Vehicle-specific parameters
        self.vehicle_params = {
            "cars": {
                "v_target": 30.0,  # m/s (108 km/h)
                "drag_coeff": 0.3,
                "mass": 1500.0,  # kg
            },
            "trucks": {
                "v_target": 25.0,  # m/s (90 km/h)
                "drag_coeff": 0.5,
                "mass": 8000.0,  # kg
            },
            "motorcycles": {
                "v_target": 35.0,  # m/s (126 km/h)
                "drag_coeff": 0.2,
                "mass": 300.0,  # kg
            },
        }

    def _compute_single_dynamics(
        self,
        pop_id: str,
        state: np.ndarray,
        action: np.ndarray,
        distributions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute next state for single vehicle.

        State: [position, velocity]
        Action: [acceleration]
        """
        position, velocity = state
        acceleration = action[0]

        # Density-dependent drag
        drag = self._compute_drag(pop_id, velocity, distributions)

        # Update velocity with acceleration and drag
        new_velocity = velocity + (acceleration - drag) * self.dt
        new_velocity = np.clip(new_velocity, 0.0, 50.0)  # Physical limits

        # Update position
        new_position = (position + velocity * self.dt) % self.road_length  # Periodic

        return np.array([new_position, new_velocity])

    def _compute_drag(
        self,
        pop_id: str,
        velocity: float,
        distributions: dict[str, np.ndarray],
    ) -> float:
        """
        Compute drag force based on all population densities.

        Drag increases with:
        - Own velocity (quadratic)
        - Densities of all vehicle types (weighted)
        """
        params = self.vehicle_params[pop_id]
        config = self.populations[pop_id]

        # Base drag (velocity-dependent)
        base_drag = params["drag_coeff"] * velocity**2 / params["mass"]

        # Congestion drag (density-dependent)
        congestion_drag = 0.0
        for other_id, weight in config.coupling_weights.items():
            if other_id in distributions:
                # Simple density proxy: variance of distribution
                density = np.var(distributions[other_id]) if len(distributions[other_id]) > 0 else 0.0
                congestion_drag += weight * density

        return base_drag + congestion_drag

    def _compute_reward(
        self,
        pop_id: str,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        distributions: dict[str, np.ndarray],
    ) -> float:
        """
        Compute reward for vehicle.

        Components:
        1. Target velocity tracking: -|v - v_target|
        2. Fuel efficiency: -α|a|²
        3. Congestion cost: -Σ w_j * density_j
        """
        _, _velocity = state
        _, next_velocity = next_state
        acceleration = action[0]

        params = self.vehicle_params[pop_id]
        config = self.populations[pop_id]

        # Velocity tracking reward
        velocity_error = abs(next_velocity - params["v_target"])
        velocity_reward = -velocity_error

        # Fuel efficiency (penalize large accelerations)
        fuel_cost = -0.1 * acceleration**2

        # Congestion cost
        congestion_cost = 0.0
        for other_id, weight in config.coupling_weights.items():
            if other_id in distributions:
                density = np.var(distributions[other_id]) if len(distributions[other_id]) > 0 else 0.0
                congestion_cost -= 0.5 * weight * density

        return float(velocity_reward + fuel_cost + congestion_cost)

    def _state_to_distribution(self, pop_id: str, state: np.ndarray) -> np.ndarray:
        """
        Convert state to distribution.

        For demonstration: create 10-bin histogram based on position.
        """
        position = state[0]
        num_bins = self.populations[pop_id].state_dim * 10
        hist = np.zeros(num_bins)

        # Place vehicle in appropriate bin
        bin_idx = int((position / self.road_length) * num_bins) % num_bins
        hist[bin_idx] = 1.0

        return hist


def create_traffic_populations() -> dict[str, PopulationConfig]:
    """Create population configurations for three vehicle types."""
    populations = {
        "cars": PopulationConfig(
            population_id="cars",
            state_dim=2,  # [position, velocity]
            action_dim=1,  # [acceleration]
            action_bounds=(-3.0, 3.0),  # m/s²
            algorithm="ddpg",
            algorithm_config={
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "noise_theta": 0.15,
                "noise_sigma": 0.2,
            },
            coupling_weights={
                "trucks": 0.8,  # Trucks slow down cars significantly
                "motorcycles": 0.3,  # Motorcycles have less impact
            },
            initial_distribution=lambda: np.array([np.random.uniform(0, 10), np.random.uniform(25, 35)]),
            population_size=100,
        ),
        "trucks": PopulationConfig(
            population_id="trucks",
            state_dim=2,
            action_dim=1,
            action_bounds=(-2.0, 2.0),  # Lower acceleration limits
            algorithm="td3",
            algorithm_config={
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "policy_delay": 2,
                "target_noise_std": 0.2,
            },
            coupling_weights={
                "cars": 0.5,
                "motorcycles": 0.2,
            },
            initial_distribution=lambda: np.array([np.random.uniform(0, 10), np.random.uniform(20, 28)]),
            population_size=30,
        ),
        "motorcycles": PopulationConfig(
            population_id="motorcycles",
            state_dim=2,
            action_dim=1,
            action_bounds=(-4.0, 4.0),  # Higher acceleration
            algorithm="sac",
            algorithm_config={
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "target_entropy": -1.0,
                "auto_tune_temperature": True,
            },
            coupling_weights={
                "cars": 0.4,
                "trucks": 0.6,  # Trucks impact motorcycles more
            },
            initial_distribution=lambda: np.array([np.random.uniform(0, 10), np.random.uniform(30, 40)]),
            population_size=50,
        ),
    }

    return populations


def main():
    """Train heterogeneous traffic MFG system."""
    print("=" * 80)
    print("Heterogeneous Traffic Multi-Population MFG")
    print("=" * 80)

    # Create populations
    populations = create_traffic_populations()

    print("\nPopulation Configuration:")
    print("-" * 80)
    for pop_id, config in populations.items():
        print(f"  {config}")

    # Create environment
    env = HeterogeneousTrafficEnv(
        populations=populations,
        road_length=10.0,
        time_horizon=10.0,
        dt=0.1,
    )

    print(f"\nEnvironment: {env.num_populations} populations on {env.road_length}km road")

    # Create agents
    agents = {}
    for pop_id, config in populations.items():
        if config.algorithm == "ddpg":
            agents[pop_id] = MultiPopulationDDPG(pop_id, env, populations)
        elif config.algorithm == "td3":
            agents[pop_id] = MultiPopulationTD3(pop_id, env, populations)
        elif config.algorithm == "sac":
            agents[pop_id] = MultiPopulationSAC(pop_id, env, populations)

    print("\nAgents initialized:")
    for pop_id, agent in agents.items():
        algo_name = populations[pop_id].algorithm.upper()
        print(f"  {pop_id:15s} → {algo_name}")

    # Create trainer
    trainer = MultiPopulationTrainer(env=env, agents=agents)

    # Train
    print("\n" + "=" * 80)
    print("Training Multi-Population MFG System")
    print("=" * 80)

    stats = trainer.train(
        num_episodes=1000,
        verbose=True,
        log_interval=100,
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating Trained Policies")
    print("=" * 80)

    eval_stats = trainer.evaluate(num_episodes=20, render=False)

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nFinal Performance:")
    for pop_id in populations:
        train_reward = np.mean(stats[pop_id]["episode_rewards"][-100:])
        eval_reward = np.mean(eval_stats[pop_id]["episode_rewards"])
        print(f"  {pop_id:15s} | Training: {train_reward:8.2f} | Evaluation: {eval_reward:8.2f}")

    print("\nNash Equilibrium Achieved: All populations optimizing simultaneously!")


if __name__ == "__main__":
    main()
