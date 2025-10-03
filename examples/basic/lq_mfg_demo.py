"""
Simple demonstration of LQ-MFG environment.

Shows basic usage of the Linear-Quadratic Mean Field Game environment:
- Environment creation and configuration
- Episode execution with random policy
- Reward tracking and visualization
- Population distribution observation

This is the simplest MFG environment for testing and validation.

Usage:
    python examples/basic/lq_mfg_demo.py
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from mfg_pde.alg.reinforcement.environments import LQMFGEnv

    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    print("Error: LQ-MFG environment not available. Install gymnasium: pip install gymnasium")


def run_random_policy_episode(env, seed: int = 42):
    """
    Run single episode with random policy.

    Args:
        env: LQ-MFG environment
        seed: Random seed for reproducibility

    Returns:
        Episode statistics (rewards, states, actions)
    """
    state, info = env.reset(seed=seed)

    episode_rewards = []
    episode_states = []
    episode_actions = []

    done = False
    truncated = False
    step = 0

    print(f"\nStarting episode (seed={seed})...")
    print(f"Initial state: x={state[0]:.2f}, v={state[1]:.2f}")

    while not (done or truncated):
        # Random action (for demonstration)
        action = env.action_space.sample()

        # Execute step
        next_state, reward, done, truncated, info = env.step(action)

        # Record
        episode_rewards.append(reward)
        episode_states.append(state.copy())
        episode_actions.append(action.copy())

        state = next_state
        step += 1

        if step % 50 == 0:
            print(f"Step {step}: x={state[0]:.2f}, v={state[1]:.2f}, reward={reward:.2f}")

    print(f"Episode finished after {step} steps")
    print(f"Final state: x={state[0]:.2f}, v={state[1]:.2f}")
    print(f"Total reward: {sum(episode_rewards):.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")

    return {
        "rewards": np.array(episode_rewards),
        "states": np.array(episode_states),
        "actions": np.array(episode_actions),
        "total_steps": step,
    }


def visualize_episode(stats: dict, save_path: str = "lq_mfg_demo.png"):
    """
    Visualize episode statistics.

    Args:
        stats: Episode statistics dictionary
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Position over time
    ax = axes[0, 0]
    positions = stats["states"][:, 0]
    ax.plot(positions, linewidth=2)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Origin")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position (x)")
    ax.set_title("Agent Position Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity over time
    ax = axes[0, 1]
    velocities = stats["states"][:, 1]
    ax.plot(velocities, linewidth=2, color="orange")
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Velocity (v)")
    ax.set_title("Agent Velocity Over Time")
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative reward
    ax = axes[1, 0]
    cumulative_rewards = np.cumsum(stats["rewards"])
    ax.plot(cumulative_rewards, linewidth=2, color="green")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.grid(True, alpha=0.3)

    # Plot 4: Control actions
    ax = axes[1, 1]
    actions = stats["actions"][:, 0]
    ax.plot(actions, linewidth=2, color="purple", alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Control (u)")
    ax.set_title("Control Actions Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def compare_cost_weights():
    """Compare behavior with different cost weights."""
    if not ENV_AVAILABLE:
        return

    print("\n" + "=" * 60)
    print("Comparing Different Cost Configurations")
    print("=" * 60)

    configs = [
        {"name": "High State Cost", "cost_state": 10.0, "cost_control": 0.1, "cost_mean_field": 0.1},
        {"name": "High Control Cost", "cost_state": 1.0, "cost_control": 1.0, "cost_mean_field": 0.1},
        {"name": "High Mean Field Cost", "cost_state": 1.0, "cost_control": 0.1, "cost_mean_field": 5.0},
    ]

    for config in configs:
        name = config.pop("name")
        env = LQMFGEnv(num_agents=50, max_steps=100, **config)

        state, _ = env.reset(seed=42)
        total_reward = 0.0

        for _ in range(100):
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        print(f"\n{name}:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final position: {state[0]:.2f}")
        print(f"  Final velocity: {state[1]:.2f}")


def main():
    """Main demonstration."""
    if not ENV_AVAILABLE:
        return

    print("=" * 60)
    print("LQ-MFG Environment Demo")
    print("=" * 60)
    print("\nLinear-Quadratic Mean Field Game:")
    print("- State: (position, velocity)")
    print("- Action: control/acceleration")
    print("- Goal: minimize distance to origin + control cost + congestion")
    print("- Dynamics: Linear (x' = x + v*dt, v' = v + u*dt)")
    print("- Cost: Quadratic (x^2 + u^2 + MF_term)")

    # Create environment
    env = LQMFGEnv(
        num_agents=50,
        x_max=10.0,
        v_max=5.0,
        u_max=2.0,
        cost_state=1.0,
        cost_control=0.1,
        cost_mean_field=0.5,
        dt=0.05,
        max_steps=200,
    )

    print("\nEnvironment configuration:")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Number of agents: {env.num_agents}")
    print(f"  Cost weights: state={env.cost_state}, control={env.cost_control}, MF={env.cost_mean_field}")

    # Run episode
    stats = run_random_policy_episode(env, seed=42)

    # Visualize
    visualize_episode(stats)

    # Compare configurations
    compare_cost_weights()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
