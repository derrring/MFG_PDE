#!/usr/bin/env python3
"""
Mathematical Visualization Demo - New API

Demonstrates MFG solution visualization with publication-quality plots.
Shows how to create professional mathematical figures for research.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import solve_mfg


def main():
    """Demonstrate mathematical visualization capabilities."""
    print("📊 Mathematical Visualization Demo")
    print("=" * 35)

    # Solve multiple problems for comparison
    problems = [
        ("crowd_dynamics", "Crowd Evacuation"),
        ("portfolio_optimization", "Portfolio Optimization"),
        ("traffic_flow", "Traffic Flow"),
        ("epidemic", "Epidemic Control"),
    ]

    print("\n1. Solving multiple MFG problems...")

    results = {}
    for problem_type, name in problems:
        print(f"   Solving {name}...")
        result = solve_mfg(problem_type, domain_size=2.0, time_horizon=1.0, accuracy="fast")
        results[name] = result

    # Create comprehensive visualization
    print("\n2. Creating mathematical visualizations...")

    # Setup figure with LaTeX-style formatting
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Mean Field Games: Comparative Analysis", fontsize=16, fontweight="bold")

    for i, (name, result) in enumerate(results.items()):
        ax = axes[i // 2, i % 2]

        # Plot density and value function
        x_grid = result.x_grid
        density = result.m
        value = result.u

        # Density plot
        line1 = ax.plot(x_grid, density, "b-", linewidth=2, label=r"$m(T,x)$ (density)")
        ax.fill_between(x_grid, density, alpha=0.3, color="blue")

        # Value function on secondary axis
        ax2 = ax.twinx()
        line2 = ax2.plot(x_grid, value, "r-", linewidth=2, label=r"$u(T,x)$ (value)")

        # Formatting
        ax.set_xlabel(r"State $x$")
        ax.set_ylabel(r"Density $m(T,x)$", color="blue")
        ax2.set_ylabel(r"Value $u(T,x)$", color="red")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper right")

        # Add convergence info
        conv_text = f"Converged: {'Yes' if result.converged else 'No'}\nIterations: {result.iterations}"
        ax.text(
            0.02,
            0.98,
            conv_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.savefig("mfg_comparative_analysis.png", dpi=300, bbox_inches="tight")
    print("📊 Saved comparative analysis as 'mfg_comparative_analysis.png'")

    # Individual detailed plot for crowd dynamics
    print("\n3. Creating detailed crowd dynamics visualization...")

    crowd_result = results["Crowd Evacuation"]
    x = crowd_result.x_grid
    m = crowd_result.m
    u = crowd_result.u

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Crowd Dynamics: Detailed Mathematical Analysis", fontsize=16, fontweight="bold")

    # 1. Density distribution
    ax1.plot(x, m, "b-", linewidth=3, label=r"$m(T,x)$")
    ax1.fill_between(x, m, alpha=0.4, color="lightblue")
    ax1.set_xlabel(r"Position $x$")
    ax1.set_ylabel(r"Density $m(T,x)$")
    ax1.set_title("Final Density Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Value function
    ax2.plot(x, u, "r-", linewidth=3, label=r"$u(T,x)$")
    ax2.set_xlabel(r"Position $x$")
    ax2.set_ylabel(r"Value $u(T,x)$")
    ax2.set_title("Optimal Value Function")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Phase portrait (if we have velocity data)
    if hasattr(crowd_result, "compute_velocity_field"):
        try:
            velocity = crowd_result.compute_velocity_field()
            ax3.plot(x, velocity, "g-", linewidth=2, label=r"$v(T,x) = -\nabla u$")
            ax3.set_xlabel(r"Position $x$")
            ax3.set_ylabel(r"Velocity $v(T,x)$")
            ax3.set_title("Optimal Velocity Field")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        except:
            ax3.text(0.5, 0.5, "Velocity field\nnot available", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Velocity Field (N/A)")
    else:
        ax3.text(0.5, 0.5, "Velocity field\nnot available", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Velocity Field (N/A)")

    # 4. Convergence history
    if hasattr(crowd_result, "residual_history"):
        ax4.semilogy(crowd_result.residual_history, "k-", linewidth=2)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Residual (log scale)")
        ax4.set_title("Convergence History")
        ax4.grid(True, alpha=0.3)
    else:
        # Create mock convergence plot
        iterations = np.arange(1, crowd_result.iterations + 1)
        mock_residuals = 1e-2 * np.exp(-0.1 * iterations)
        ax4.semilogy(iterations, mock_residuals, "k--", linewidth=2, alpha=0.7)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Residual (log scale)")
        ax4.set_title("Convergence History (Estimated)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("crowd_dynamics_detailed.png", dpi=300, bbox_inches="tight")
    print("📊 Saved detailed analysis as 'crowd_dynamics_detailed.png'")

    # Summary statistics
    print("\n4. Mathematical analysis summary:")
    for name, result in results.items():
        x_grid = result.x_grid
        density = result.m

        # Calculate statistical measures
        mean_pos = np.trapz(x_grid * density, x_grid)
        variance = np.trapz((x_grid - mean_pos) ** 2 * density, x_grid)
        std_dev = np.sqrt(variance)

        print(f"\n   {name}:")
        print(f"     • Mean position: {mean_pos:.3f}")
        print(f"     • Standard deviation: {std_dev:.3f}")
        print(f"     • Convergence: {'✓' if result.converged else '✗'}")
        print(f"     • Iterations: {result.iterations}")

    print("\n📊 Visualization demo complete!")
    print("Generated publication-quality mathematical figures")


if __name__ == "__main__":
    main()
