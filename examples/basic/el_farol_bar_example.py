#!/usr/bin/env python3
"""
El Farol Bar Problem - Simple New API Demo

Classic economic problem: agents decide whether to attend a bar where
enjoyment decreases with crowding. Demonstrates coordination in MFG.

Mathematical formulation:
- Agents choose attendance probability x ∈ [0,1]
- Higher attendance creates congestion costs
- Equilibrium balances individual incentives with crowd effects
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import solve_mfg


def main():
    """Demonstrate El Farol bar problem using new API."""
    print("🍺 El Farol Bar Problem - MFG Demo")
    print("=" * 40)

    # Quick solution using simple API
    print("\n1. Quick solution with built-in crowd dynamics:")
    result = solve_mfg(
        "crowd_dynamics",
        domain_size=1.0,  # Attendance probability [0,1]
        crowd_size=100,  # Number of agents
        time_horizon=1.0,  # Decision time scale
        accuracy="fast",
    )

    print(f"✅ Converged: {result.converged}")
    print(f"📊 Iterations: {result.iterations}")

    # Analyze the equilibrium
    print("\n2. Economic analysis:")
    final_density = result.m  # Population density
    x_grid = result.x_grid  # Spatial grid

    # Expected attendance rate
    expected_attendance = float(np.trapezoid(x_grid * final_density, x_grid))
    print(f"📈 Expected attendance rate: {expected_attendance:.2%}")

    # Distribution analysis
    home_bias = float(np.trapezoid(final_density[x_grid < 0.5], x_grid[x_grid < 0.5]))
    bar_bias = float(np.trapezoid(final_density[x_grid >= 0.5], x_grid[x_grid >= 0.5]))

    print(f"🏠 Home-biased agents: {home_bias:.1%}")
    print(f"🍺 Bar-biased agents: {bar_bias:.1%}")

    # Visualize the results
    print("\n3. Creating visualization...")

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Density distribution
    ax1.plot(x_grid, final_density, "b-", linewidth=2, label="Agent density")
    ax1.fill_between(x_grid, final_density, alpha=0.3)
    ax1.set_xlabel("Attendance tendency")
    ax1.set_ylabel("Population density")
    ax1.set_title("Equilibrium Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Value function
    value_function = result.u
    ax2.plot(x_grid, value_function, "r-", linewidth=2, label="Value function")
    ax2.set_xlabel("Attendance tendency")
    ax2.set_ylabel("Expected cost")
    ax2.set_title("Optimal Value Function")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("el_farol_equilibrium.png", dpi=150, bbox_inches="tight")
    print("📊 Saved plot as 'el_farol_equilibrium.png'")

    # Economic interpretation
    print("\n4. Economic insights:")
    print(f"   • Attendance rate {expected_attendance:.1%} represents the equilibrium")
    print(f"   • {home_bias:.0%} of agents lean toward staying home")
    print(f"   • {bar_bias:.0%} of agents lean toward going to bar")
    print("   • The distribution reflects how crowding concerns affect decisions")

    # Parameter study
    print("\n5. Parameter sensitivity:")
    crowd_sizes = [50, 100, 200, 300]
    attendance_rates = []

    for crowd_size in crowd_sizes:
        result = solve_mfg("crowd_dynamics", domain_size=1.0, crowd_size=crowd_size, time_horizon=1.0, accuracy="fast")

        attendance = float(np.trapezoid(result.x_grid * result.m, result.x_grid))
        attendance_rates.append(attendance)
        print(f"   • {crowd_size} agents → {attendance:.1%} attendance")

    plt.figure(figsize=(8, 5))
    plt.plot(crowd_sizes, [rate * 100 for rate in attendance_rates], "o-", linewidth=2)
    plt.xlabel("Population size")
    plt.ylabel("Attendance rate (%)")
    plt.title("Crowd Size vs Attendance Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig("el_farol_sensitivity.png", dpi=150, bbox_inches="tight")
    print("📊 Saved sensitivity plot as 'el_farol_sensitivity.png'")

    print("\n🎯 El Farol demonstration complete!")
    print("Key insight: Larger populations lead to higher coordination efficiency")


if __name__ == "__main__":
    main()
