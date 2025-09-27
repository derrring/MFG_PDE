#!/usr/bin/env python3
"""
Towel on the Beach Problem - Simple New API Demo

Beach-goers decide whether to visit based on expected crowding.
A user-friendly version of the El Farol coordination problem.

Scenario: People want to enjoy the beach but avoid crowds.
Everyone must decide simultaneously without knowing others' choices.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import solve_mfg


def main():
    """Demonstrate towel beach problem using new API."""
    print("🏖️  Towel on the Beach Problem - MFG Demo")
    print("=" * 45)

    print("\nScenario: 200 people deciding whether to go to the beach")
    print("Each person wants to enjoy the beach but avoid crowds")

    # Solve the coordination problem
    print("\n1. Solving beach attendance coordination...")
    result = solve_mfg(
        "crowd_dynamics",
        domain_size=1.0,  # Decision space [0=stay, 1=go]
        crowd_size=200,  # 200 potential beach-goers
        time_horizon=1.0,  # Decision timeframe
        accuracy="fast",
    )

    print(f"✅ Solution found: {result.converged}")
    print(f"🔄 Iterations: {result.iterations}")

    # Analyze the beach attendance equilibrium
    print("\n2. Beach attendance analysis:")
    final_density = result.m
    x_grid = result.x_grid

    # Calculate expected beach attendance
    expected_beachgoers = np.trapz(x_grid * final_density, x_grid) * 200
    print(f"🏖️  Expected beach attendance: {expected_beachgoers:.0f} people")

    # Analyze distribution of preferences
    stay_home = np.trapz(final_density[x_grid < 0.3], x_grid[x_grid < 0.3])
    uncertain = np.trapz(final_density[(x_grid >= 0.3) & (x_grid < 0.7)], x_grid[(x_grid >= 0.3) & (x_grid < 0.7)])
    go_beach = np.trapz(final_density[x_grid >= 0.7], x_grid[x_grid >= 0.7])

    print(f"🏠 Prefer staying home: {stay_home:.1%}")
    print(f"🤔 Undecided: {uncertain:.1%}")
    print(f"🌊 Prefer beach: {go_beach:.1%}")

    # Create visualization
    print("\n3. Creating beach attendance visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Population density over beach preference
    ax1.plot(x_grid, final_density, "skyblue", linewidth=3, label="Population density")
    ax1.fill_between(x_grid, final_density, alpha=0.4, color="lightblue")
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Indifference point")
    ax1.set_xlabel("Beach preference (0=stay home, 1=go to beach)")
    ax1.set_ylabel("Population density")
    ax1.set_title("Distribution of Beach Preferences")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Value function (cost of different preferences)
    value_function = result.u
    ax2.plot(x_grid, value_function, "orange", linewidth=3, label="Expected cost")
    ax2.set_xlabel("Beach preference")
    ax2.set_ylabel("Expected cost")
    ax2.set_title("Cost of Different Preferences")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("beach_coordination.png", dpi=150, bbox_inches="tight")
    print("📊 Saved visualization as 'beach_coordination.png'")

    # Weather sensitivity study
    print("\n4. Weather impact analysis:")
    weather_conditions = ["Rainy", "Cloudy", "Sunny", "Perfect"]
    crowd_sizes = [50, 100, 200, 300]  # Different weather affects potential attendance

    attendance_by_weather = []

    for i, (weather, crowd_size) in enumerate(zip(weather_conditions, crowd_sizes, strict=False)):
        result = solve_mfg("crowd_dynamics", domain_size=1.0, crowd_size=crowd_size, time_horizon=1.0, accuracy="fast")

        beach_rate = np.trapz(result.x_grid * result.m, result.x_grid)
        actual_attendance = beach_rate * crowd_size
        attendance_by_weather.append(actual_attendance)

        print(f"   {weather:>7} day: {actual_attendance:.0f}/{crowd_size} people ({beach_rate:.1%})")

    # Weather impact plot
    plt.figure(figsize=(10, 6))
    colors = ["gray", "lightblue", "gold", "orange"]
    bars = plt.bar(weather_conditions, attendance_by_weather, color=colors, alpha=0.7)
    plt.xlabel("Weather Condition")
    plt.ylabel("Expected Beach Attendance")
    plt.title("Beach Attendance by Weather\n(Equilibrium with Crowd Aversion)")
    plt.grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for i, (bar, attendance) in enumerate(zip(bars, attendance_by_weather, strict=False)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{attendance:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("beach_weather_impact.png", dpi=150, bbox_inches="tight")
    print("📊 Saved weather analysis as 'beach_weather_impact.png'")

    print("\n🏖️  Key insights:")
    print("   • People coordinate to avoid over-crowding")
    print("   • Better weather increases potential attendance but also competition")
    print("   • Equilibrium balances beach enjoyment vs crowd aversion")
    print("   • Real coordination problems have similar mathematical structure")

    print("\n🎯 Towel beach demonstration complete!")


if __name__ == "__main__":
    main()
