#!/usr/bin/env python3
"""
Traffic Flow Mean Field Game: Multi-Vehicle Type Coordination.

This example demonstrates MFG-RL applied to realistic traffic scenarios with:
- Multiple vehicle types (cars, trucks, buses)
- Different speeds, lane restrictions, and objectives
- Congestion-aware routing
- Nash equilibrium traffic patterns

Application Domain: Autonomous Vehicle Coordination
Key Features:
- Heterogeneous vehicle capabilities
- Strategic lane changing and routing
- Population-dependent travel times (congestion)
- Emergent traffic flow patterns

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


def create_traffic_network():
    """
    Create road network represented as a grid.

    Layout represents a simplified urban grid:
    - Vertical roads (north-south arterials)
    - Horizontal roads (east-west streets)
    - Intersections
    """
    # 20x20 grid representing road network
    # 0 = road, 1 = building/obstacle
    network = np.ones((20, 20), dtype=np.int32)

    # Major arterial roads (vertical) - 3 lanes wide
    network[:, 3:6] = 0  # West arterial
    network[:, 9:12] = 0  # Central arterial
    network[:, 15:18] = 0  # East arterial

    # Cross streets (horizontal) - 2 lanes wide
    network[3:5, :] = 0  # North street
    network[9:11, :] = 0  # Central street
    network[15:17, :] = 0  # South street

    return network


def create_traffic_environment():
    """Create multi-vehicle traffic environment."""
    network = create_traffic_network()

    # Car configuration (fast, flexible)
    car_config = AgentTypeConfig(
        type_id="car",
        type_index=0,
        action_type=ActionType.FOUR_CONNECTED,
        speed_multiplier=1.0,  # Standard speed
        reward_type=RewardType.CONGESTION,
        goal_reward=100.0,  # High reward for reaching destination
        collision_penalty=-5.0,
        move_cost=0.1,  # Fuel cost
        congestion_weight=2.0,  # Strongly avoid congestion
        cross_population_weights={
            "truck": 1.5,  # Avoid trucks (slower, harder to pass)
            "bus": 1.0,  # Avoid buses
        },
        start_positions=[(1, 4), (1, 10), (1, 16)],  # North edge entrances
        goal_positions=[(18, 4), (18, 10), (18, 16)],  # South edge exits
        num_agents=8,
    )

    # Truck configuration (slower, restricted lanes)
    truck_config = AgentTypeConfig(
        type_id="truck",
        type_index=1,
        action_type=ActionType.FOUR_CONNECTED,
        speed_multiplier=0.7,  # Slower than cars
        reward_type=RewardType.CONGESTION,
        goal_reward=80.0,
        collision_penalty=-5.0,
        move_cost=0.15,  # Higher fuel cost
        congestion_weight=1.0,  # Less sensitive to congestion
        cross_population_weights={
            "car": 0.5,  # Cars less problematic for trucks
            "bus": 0.8,
        },
        start_positions=[(1, 4), (1, 10)],  # Only use arterials
        goal_positions=[(18, 4), (18, 10)],
        num_agents=4,
    )

    # Bus configuration (scheduled route, moderate speed)
    bus_config = AgentTypeConfig(
        type_id="bus",
        type_index=2,
        action_type=ActionType.FOUR_CONNECTED,
        speed_multiplier=0.85,
        reward_type=RewardType.MFG_STANDARD,  # Time-based (schedule adherence)
        goal_reward=120.0,  # High priority to stay on schedule
        collision_penalty=-10.0,  # Must avoid collisions
        move_cost=0.2,
        congestion_weight=1.5,  # Moderate congestion sensitivity
        cross_population_weights={
            "car": 0.3,
            "truck": 1.2,  # Trucks slow buses down
        },
        start_positions=[(1, 10)],  # Central arterial only
        goal_positions=[(18, 10)],
        num_agents=3,
    )

    # Multi-vehicle configuration
    config = MultiPopulationMazeConfig(
        maze_array=network,
        agent_types={"car": car_config, "truck": truck_config, "bus": bus_config},
        population_smoothing=0.3,
        population_update_frequency=5,
        max_episode_steps=300,
        time_penalty=-0.01,  # Incentive to reach destination quickly
    )

    return MultiPopulationMazeEnvironment(config)


def train_traffic_coordination(env, solvers, num_iterations=15, episodes_per_iteration=40):
    """
    Train traffic coordination using alternating best-response.

    Simulates traffic learning dynamics where:
    1. Cars learn optimal routes given truck/bus patterns
    2. Trucks learn optimal routes given car/bus patterns
    3. Buses learn optimal routes given car/truck patterns
    """
    print("=" * 80)
    print("Traffic Flow MFG: Multi-Vehicle Coordination Training")
    print("=" * 80)

    car_solver = solvers["car"]
    truck_solver = solvers["truck"]
    bus_solver = solvers["bus"]

    history = {
        "car_rewards": [],
        "truck_rewards": [],
        "bus_rewards": [],
    }

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Traffic Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*80}")

        # Train cars
        print("\n[Cars] Learning optimal routes...")
        car_solver.epsilon = max(0.2, car_solver.epsilon)
        car_results = car_solver.train(num_episodes=episodes_per_iteration)
        history["car_rewards"].extend(car_results["episode_rewards"])
        print(f"Car avg reward (last 20): " f"{np.mean(car_results['episode_rewards'][-20:]):.2f}")

        # Train trucks
        print("\n[Trucks] Learning optimal routes...")
        truck_solver.epsilon = max(0.2, truck_solver.epsilon)
        truck_results = truck_solver.train(num_episodes=episodes_per_iteration)
        history["truck_rewards"].extend(truck_results["episode_rewards"])
        print(f"Truck avg reward (last 20): " f"{np.mean(truck_results['episode_rewards'][-20:]):.2f}")

        # Train buses
        print("\n[Buses] Learning optimal routes...")
        bus_solver.epsilon = max(0.2, bus_solver.epsilon)
        bus_results = bus_solver.train(num_episodes=episodes_per_iteration)
        history["bus_rewards"].extend(bus_results["episode_rewards"])
        print(f"Bus avg reward (last 20): " f"{np.mean(bus_results['episode_rewards'][-20:]):.2f}")

        # Summary
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1} Summary:")
        print(f"  Cars:   {np.mean(car_results['episode_rewards']):.2f}")
        print(f"  Trucks: {np.mean(truck_results['episode_rewards']):.2f}")
        print(f"  Buses:  {np.mean(bus_results['episode_rewards']):.2f}")
        print(f"{'='*80}")

    return history


def visualize_training(history):
    """Plot training progress for all vehicle types."""
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vehicle_types = ["car", "truck", "bus"]
    colors = ["blue", "red", "green"]
    titles = ["Cars", "Trucks", "Buses"]

    for idx, (vtype, color, title) in enumerate(zip(vehicle_types, colors, titles, strict=False)):
        ax = axes[idx]
        rewards = history[f"{vtype}_rewards"]

        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=title, color=color, linewidth=2)
        else:
            ax.plot(rewards, label=title, color=color, linewidth=2)

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total Reward (smoothed)", fontsize=12)
        ax.set_title(f"{title} Learning Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("traffic_flow_training.png", dpi=150, bbox_inches="tight")
    print("\nSaved training plot to: traffic_flow_training.png")


def visualize_traffic_flow(env):
    """Visualize traffic density distributions."""
    # Run episode to get traffic patterns
    _observations, _ = env.reset(seed=42)

    for _ in range(150):
        actions = {
            "car": np.random.randint(0, 4, size=8),
            "truck": np.random.randint(0, 4, size=4),
            "bus": np.random.randint(0, 4, size=3),
        }
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break

    multi_pop_state = env.get_multi_population_state()
    densities = multi_pop_state.get_all_densities()

    _fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Individual vehicle type densities
    vehicle_types = ["car", "truck", "bus"]
    colors = ["Blues", "Reds", "Greens"]
    titles = ["Car Density", "Truck Density", "Bus Density"]

    for idx, (vtype, cmap, title) in enumerate(zip(vehicle_types, colors, titles, strict=False)):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(densities[vtype], cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Column (East-West)")
        ax.set_ylabel("Row (North-South)")
        plt.colorbar(im, ax=ax, label="Vehicle Density")

        # Overlay road network
        network = env.maze_array
        ax.contour(network, levels=[0.5], colors="black", linewidths=0.5, alpha=0.3)

    # Combined traffic flow
    ax = axes[1, 1]
    overlay = np.zeros((*densities["car"].shape, 3))
    overlay[:, :, 2] = densities["car"]  # Blue channel (cars)
    overlay[:, :, 0] = densities["truck"]  # Red channel (trucks)
    overlay[:, :, 1] = densities["bus"]  # Green channel (buses)
    overlay = overlay / overlay.max() if overlay.max() > 0 else overlay

    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("Combined Traffic Flow\n(Blue=Cars, Red=Trucks, Green=Buses)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column (East-West)")
    ax.set_ylabel("Row (North-South)")

    # Overlay road network
    ax.contour(network, levels=[0.5], colors="white", linewidths=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig("traffic_flow_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved distribution plot to: traffic_flow_distribution.png")


def main():
    """Run traffic flow MFG application."""
    print("=" * 80)
    print("Traffic Flow Mean Field Game - Multi-Vehicle Coordination")
    print("=" * 80)
    print("\nApplication:")
    print("  - Autonomous vehicle coordination in urban network")
    print("  - 3 vehicle types: Cars (fast, flexible), Trucks (slow, restricted),")
    print("    Buses (scheduled routes)")
    print("\nObjective:")
    print("  - Learn Nash equilibrium traffic patterns")
    print("  - Minimize travel time while avoiding congestion")
    print("  - Emergent lane selection and routing strategies")

    # Create environment
    env = create_traffic_environment()

    print("\nRoad Network:")
    print(f"  Size: {env.maze_array.shape}")
    print(f"  Cars: {env.agent_types['car'].num_agents}")
    print(f"  Trucks: {env.agent_types['truck'].num_agents}")
    print(f"  Buses: {env.agent_types['bus'].num_agents}")

    # Get dimensions
    obs, _ = env.reset(seed=42)

    state_dims = {
        "car": obs["car"].shape[1],
        "truck": obs["truck"].shape[1],
        "bus": obs["bus"].shape[1],
    }

    action_dims = {
        "car": env.action_spaces["car"].n,
        "truck": env.action_spaces["truck"].n,
        "bus": env.action_spaces["bus"].n,
    }

    height, width = env.maze_array.shape
    population_dims = {
        "car": height * width,
        "truck": height * width,
        "bus": height * width,
    }

    print("\nNetwork Architecture:")
    print(f"  Car state: {state_dims['car']}, actions: {action_dims['car']}")
    print(f"  Truck state: {state_dims['truck']}, actions: {action_dims['truck']}")
    print(f"  Bus state: {state_dims['bus']}, actions: {action_dims['bus']}")

    # Create solvers
    print("\nInitializing multi-vehicle Q-learning solvers...")
    solvers = create_multi_population_q_learning_solvers(
        env=env,
        state_dims=state_dims,
        action_dims=action_dims,
        population_dims=population_dims,
        config={
            "learning_rate": 3e-4,
            "discount_factor": 0.95,
            "epsilon": 1.0,
            "epsilon_decay": 0.98,
            "epsilon_min": 0.1,
            "batch_size": 64,
            "target_update_frequency": 50,
        },
    )

    # Train
    print("\nTraining traffic coordination...")
    history = train_traffic_coordination(env=env, solvers=solvers, num_iterations=15, episodes_per_iteration=40)

    # Visualize
    print("\n" + "=" * 80)
    print("Training Complete - Generating Visualizations")
    print("=" * 80)

    visualize_training(history)
    visualize_traffic_flow(env)

    # Summary
    print("\n" + "=" * 80)
    print("Traffic Flow Analysis")
    print("=" * 80)

    car_final = np.mean(history["car_rewards"][-100:])
    truck_final = np.mean(history["truck_rewards"][-100:])
    bus_final = np.mean(history["bus_rewards"][-100:])

    print("\nFinal Performance (last 100 episodes):")
    print(f"  Cars:   {car_final:.2f}")
    print(f"  Trucks: {truck_final:.2f}")
    print(f"  Buses:  {bus_final:.2f}")

    print("\nEmergent Behaviors:")
    if car_final > 50:
        print("  ✓ Cars learned efficient routing with congestion avoidance")
    if truck_final > 40:
        print("  ✓ Trucks learned to use arterial roads efficiently")
    if bus_final > 60:
        print("  ✓ Buses learned schedule-optimal routes")

    print("\nKey Insights:")
    print("  - Heterogeneous vehicle types create complex traffic dynamics")
    print("  - Nash equilibrium emerges from strategic route selection")
    print("  - Congestion awareness leads to self-organized traffic flow")
    print("  - Multi-population MFG captures realistic urban traffic patterns")

    print("\nReal-World Applications:")
    print("  - Autonomous vehicle coordination systems")
    print("  - Traffic signal optimization")
    print("  - Dynamic routing and navigation")
    print("  - Urban transportation planning")

    print("\n" + "=" * 80)
    print("Traffic Flow MFG Application Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
