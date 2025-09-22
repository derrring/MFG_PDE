#!/usr/bin/env python3
"""
Towel on the Beach / Beach Bar Process - Spatial MFG Implementation

This implements the correct spatial competition model where agents choose positions
on a beach to balance proximity to an ice cream stall against crowd avoidance.

Mathematical Formulation:
- State space: x ∈ [0,1] (position on beach)
- Ice cream stall at position x_stall
- Running cost: L(x,u,m) = |x - x_stall| + ln(m(x,t)) + (1/2)u²
- Agents seek proximity to stall while avoiding crowded areas

This is DISTINCT from the Santa Fe El Farol Bar problem, which deals with
attendance decisions rather than spatial positioning.

Reference: arXiv:2007.03458 "Beach Bar Process"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.integration import trapezoid
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("towel_beach_spatial", level="INFO")
logger = get_logger(__name__)


class TowelBeachSpatialProblem(MFGProblem):
    """
    Towel on the Beach Problem: Spatial Competition MFG

    Models agents choosing positions on a beach to balance:
    1. Proximity to ice cream stall (attraction)
    2. Avoidance of crowded areas (repulsion)

    This creates spatial sorting with equilibrium patterns ranging from
    single peaks (low crowd aversion) to "crater" patterns (high crowd aversion).
    """

    def __init__(self, stall_position=0.5, crowd_aversion=1.0, movement_cost=0.5, noise_level=0.1, **kwargs):
        """
        Initialize the Towel Beach Spatial Problem.

        Args:
            stall_position: Location of ice cream stall on [0,1]
            crowd_aversion: Strength of congestion penalty (ln(m) coefficient)
            movement_cost: Cost of changing position (u² coefficient)
            noise_level: Diffusion strength (σ)
            **kwargs: MFG problem parameters
        """
        super().__init__(**kwargs)
        self.stall_position = stall_position
        self.crowd_aversion = crowd_aversion
        self.movement_cost = movement_cost
        self.noise_level = noise_level

        logger.info("Created Towel Beach Spatial Problem:")
        logger.info(f"  Stall position: {stall_position}")
        logger.info(f"  Crowd aversion: {crowd_aversion}")
        logger.info(f"  Movement cost: {movement_cost}")
        logger.info(f"  Noise level: {noise_level}")

    def g(self, x):
        """
        Terminal cost: penalty for final distance from stall.

        At the end of the time horizon, agents prefer to be near the stall.
        """
        return 0.5 * np.abs(x - self.stall_position)

    def f(self, x, u, m):
        """
        Running cost: L(x,u,m) = |x - x_stall| + λ*ln(m(x,t)) + (1/2)u²

        Args:
            x: Current position on beach [0,1]
            u: Control (velocity/movement rate)
            m: Population density at current position

        Returns:
            Instantaneous cost for being at position x with control u
        """
        # Distance penalty: farther from stall = higher cost
        distance_cost = np.abs(x - self.stall_position)

        # Congestion penalty: logarithmic penalty for crowd density
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        congestion_cost = self.crowd_aversion * np.log(np.maximum(m, epsilon))

        # Movement cost: quadratic penalty for velocity
        movement_cost = self.movement_cost * u**2

        return distance_cost + congestion_cost + movement_cost

    def rho0(self, x):
        """
        Initial population distribution.

        Start with agents uniformly distributed across the beach,
        with slight concentration around the middle.
        """
        # Gaussian distribution centered slightly off-center for asymmetry
        center = 0.4
        width = 0.3
        density = np.exp(-(((x - center) / width) ** 2))

        # Normalize to integrate to 1
        return density / trapezoid(density, dx=self.Dx)

    def sigma(self, x):
        """Noise level in agent movement."""
        return self.noise_level

    def boundary_conditions_m(self):
        """Population density boundary conditions: no-flux (agents don't leave beach)."""
        return BoundaryConditions(type="no_flux")

    def boundary_conditions_u(self):
        """Value function boundary conditions: Neumann (derivative = 0)."""
        return BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)

    def H(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """
        Hamiltonian: H(x,p,m) for optimal control computation.

        From the running cost L(x,u,m) = |x - x_stall| + λ*ln(m) + (1/2)u²,
        the Hamiltonian includes the optimized control term.
        """
        x = self.xSpace[x_idx]

        # Get spatial derivative of value function (costate)
        p_x = p_values.get("p_x", 0.0)

        # Optimal control: u* = -p_x (from ∂H/∂u = 0)
        u_optimal = -p_x

        # Distance cost
        distance_cost = np.abs(x - self.stall_position)

        # Congestion cost
        epsilon = 1e-6
        congestion_cost = self.crowd_aversion * np.log(np.maximum(m_at_x, epsilon))

        # Hamiltonian: L + p_x * u* = L - (1/2)*p_x²
        return distance_cost + congestion_cost - 0.5 * self.movement_cost * p_x**2

    def dH_dm(self, x_idx: int, m_at_x: float, p_values: dict, t_idx=None):
        """
        Derivative of Hamiltonian with respect to population density.

        ∂H/∂m = ∂/∂m[λ*ln(m)] = λ/m
        """
        epsilon = 1e-6
        return self.crowd_aversion / np.maximum(m_at_x, epsilon)

    def analyze_equilibrium(self, U, M):
        """
        Analyze the spatial equilibrium pattern.

        Returns analysis of the population distribution and key metrics.
        """
        logger.info("Analyzing spatial equilibrium pattern...")

        x_grid = self.xSpace
        final_density = M[-1, :]

        # Find peaks and valleys in density
        density_gradient = np.gradient(final_density, x_grid)
        critical_points = []
        for i in range(1, len(density_gradient) - 1):
            if (density_gradient[i - 1] > 0 and density_gradient[i + 1] < 0) or (
                density_gradient[i - 1] < 0 and density_gradient[i + 1] > 0
            ):
                critical_points.append((x_grid[i], final_density[i]))

        # Density at stall location
        stall_idx = np.argmin(np.abs(x_grid - self.stall_position))
        density_at_stall = final_density[stall_idx]

        # Maximum density and its location
        max_density_idx = np.argmax(final_density)
        max_density_location = x_grid[max_density_idx]
        max_density_value = final_density[max_density_idx]

        # Spatial spread (standard deviation)
        mean_position = trapezoid(x_grid * final_density, x_grid)
        variance = trapezoid((x_grid - mean_position) ** 2 * final_density, x_grid)
        spatial_spread = np.sqrt(variance)

        # Classify equilibrium type
        equilibrium_type = self._classify_equilibrium(final_density, stall_idx)

        analysis = {
            "equilibrium_type": equilibrium_type,
            "critical_points": critical_points,
            "density_at_stall": density_at_stall,
            "max_density_location": max_density_location,
            "max_density_value": max_density_value,
            "mean_position": mean_position,
            "spatial_spread": spatial_spread,
            "stall_position": self.stall_position,
            "crowd_aversion": self.crowd_aversion,
        }

        logger.info(f"Equilibrium type: {equilibrium_type}")
        logger.info(f"Density at stall: {density_at_stall:.3f}")
        logger.info(f"Max density at x={max_density_location:.3f}: {max_density_value:.3f}")
        logger.info(f"Spatial spread: {spatial_spread:.3f}")

        return analysis

    def _classify_equilibrium(self, density, stall_idx):
        """Classify the type of spatial equilibrium."""
        density_at_stall = density[stall_idx]
        max_density = np.max(density)

        # Check if stall is the densest location
        if density_at_stall >= 0.9 * max_density:
            return "Single Peak (Low Crowd Aversion)"

        # Check for crater pattern (low density at stall, peaks on sides)
        left_peak = np.max(density[:stall_idx]) if stall_idx > 0 else 0
        right_peak = np.max(density[stall_idx + 1 :]) if stall_idx < len(density) - 1 else 0

        if (left_peak > 1.2 * density_at_stall) or (right_peak > 1.2 * density_at_stall):
            return "Crater Pattern (High Crowd Aversion)"

        return "Mixed Pattern (Moderate Crowd Aversion)"


def solve_beach_spatial_variants():
    """
    Solve the beach spatial problem with different crowd aversion parameters
    to demonstrate the transition from single peak to crater equilibria.
    """
    logger.info("Solving Beach Spatial Problem with varying crowd aversion")

    scenarios = [
        {"name": "Low Aversion", "crowd_aversion": 0.5, "description": "Weak crowding penalty"},
        {"name": "Medium Aversion", "crowd_aversion": 1.5, "description": "Moderate crowding penalty"},
        {"name": "High Aversion", "crowd_aversion": 3.0, "description": "Strong crowding penalty"},
    ]

    results = {}

    for scenario in scenarios:
        logger.info(f"\\nSolving {scenario['name']} scenario...")

        # Create problem
        problem = TowelBeachSpatialProblem(
            T=1.0,
            Nx=100,
            Nt=50,
            stall_position=0.6,  # Slightly off-center for asymmetry
            crowd_aversion=scenario["crowd_aversion"],
            movement_cost=0.5,
            noise_level=0.1,
        )

        # Solve with fast solver
        solver = create_fast_solver(problem)
        result = solver.solve()
        U, M = result.U, result.M
        info = result.metadata

        # Analyze equilibrium
        equilibrium_analysis = problem.analyze_equilibrium(U, M)

        results[scenario["name"]] = {
            "problem": problem,
            "solution": (U, M),
            "info": info,
            "analysis": equilibrium_analysis,
            "crowd_aversion": scenario["crowd_aversion"],
            "description": scenario["description"],
        }

        logger.info(f"  Equilibrium: {equilibrium_analysis['equilibrium_type']}")
        logger.info(f"  Spatial spread: {equilibrium_analysis['spatial_spread']:.3f}")
        logger.info(f"  Converged: {info.get('converged', 'Unknown')}")

    return results


def create_spatial_visualization(results):
    """Create comprehensive visualization of spatial equilibrium patterns."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Towel on Beach: Spatial Competition with Varying Crowd Aversion", fontsize=16, fontweight="bold")

    colors = ["blue", "green", "red"]
    scenario_names = list(results.keys())

    # Plot 1: Final Density Distributions
    ax1 = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        problem = result["problem"]
        U, M = result["solution"]
        x_grid = problem.xSpace

        ax1.plot(x_grid, M[-1, :], color=colors[i], linewidth=2, label=f"{name} (λ={result['crowd_aversion']})")

        # Mark stall position
        ax1.axvline(x=problem.stall_position, color=colors[i], linestyle="--", alpha=0.5)

    ax1.set_xlabel("Beach Position")
    ax1.set_ylabel("Population Density")
    ax1.set_title("Final Spatial Distributions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Density Evolution Over Time (Medium Aversion)
    ax2 = axes[0, 1]
    medium_result = results["Medium Aversion"]
    problem = medium_result["problem"]
    U, M = medium_result["solution"]

    X, T = np.meshgrid(problem.xSpace, np.linspace(0, problem.T, problem.Nt))
    contour = ax2.contourf(X, T, M, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax2, label="Density")
    ax2.axvline(x=problem.stall_position, color="red", linestyle="--", alpha=0.8, label="Stall")
    ax2.set_xlabel("Beach Position")
    ax2.set_ylabel("Time")
    ax2.set_title("Density Evolution (Medium Aversion)")
    ax2.legend()

    # Plot 3: Equilibrium Type Analysis
    ax3 = axes[0, 2]
    crowd_aversions = [result["crowd_aversion"] for result in results.values()]
    densities_at_stall = [result["analysis"]["density_at_stall"] for result in results.values()]
    max_densities = [result["analysis"]["max_density_value"] for result in results.values()]

    ax3.plot(crowd_aversions, densities_at_stall, "ro-", linewidth=2, markersize=8, label="Density at Stall")
    ax3.plot(crowd_aversions, max_densities, "bs-", linewidth=2, markersize=8, label="Maximum Density")
    ax3.set_xlabel("Crowd Aversion Parameter (λ)")
    ax3.set_ylabel("Density")
    ax3.set_title("Stall vs Maximum Density")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spatial Spread Analysis
    ax4 = axes[1, 0]
    spatial_spreads = [result["analysis"]["spatial_spread"] for result in results.values()]

    bars = ax4.bar(scenario_names, spatial_spreads, color=colors, alpha=0.7)
    ax4.set_ylabel("Spatial Spread (σ)")
    ax4.set_title("Population Dispersion")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, spread in zip(bars, spatial_spreads, strict=False):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{spread:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 5: Equilibrium Classification
    ax5 = axes[1, 1]
    ax5.axis("off")

    classification_text = "EQUILIBRIUM CLASSIFICATION\\n\\n"
    for name, result in results.items():
        analysis = result["analysis"]
        classification_text += f"{name}:\\n"
        classification_text += f"  Type: {analysis['equilibrium_type']}\\n"
        classification_text += f"  Stall density: {analysis['density_at_stall']:.3f}\\n"
        classification_text += f"  Max density: {analysis['max_density_value']:.3f}\\n"
        classification_text += f"  Spread: {analysis['spatial_spread']:.3f}\\n\\n"

    ax5.text(
        0.05,
        0.95,
        classification_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # Plot 6: Problem Description
    ax6 = axes[1, 2]
    ax6.axis("off")

    description_text = """
TOWEL ON BEACH SPATIAL MODEL

Mathematical Formulation:
• Running cost: L(x,u,m) = |x-x_stall| + λ·ln(m) + ½u²
• State: x ∈ [0,1] (position on beach)
• Control: u (movement velocity)
• Stall location: x_stall (attraction point)

Key Parameters:
• λ (crowd aversion): Controls congestion penalty
• Ice cream stall: Fixed attraction point
• Spatial competition: Position choice problem

Equilibrium Types:
• Low λ → Single peak at stall
• High λ → Crater pattern (avoid stall area)
• Spatial sorting emerges naturally

Physical Interpretation:
Agents balance proximity to amenity
against overcrowding costs, leading
to rich spatial equilibrium patterns.
    """

    ax6.text(
        0.05,
        0.95,
        description_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    output_dir = Path("examples/basic/outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "towel_beach_spatial_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Spatial visualization saved to: {output_path}")

    return output_path


def main():
    """Main execution function for spatial beach problem."""
    logger.info("Starting Towel on Beach Spatial Problem Analysis")

    print("🏖️  TOWEL ON THE BEACH - SPATIAL COMPETITION MODEL")
    print("=" * 60)
    print("Mathematical analysis of spatial positioning under congestion")
    print()
    print("Problem: Agents choose beach positions balancing:")
    print("  • Proximity to ice cream stall (attraction)")
    print("  • Avoidance of crowded areas (repulsion)")
    print()
    print("This demonstrates spatial sorting in Mean Field Games")
    print()

    try:
        # Solve different scenarios
        results = solve_beach_spatial_variants()

        # Create visualization
        plot_path = create_spatial_visualization(results)

        # Display results summary
        print("\\n SPATIAL EQUILIBRIUM ANALYSIS")
        print("-" * 40)
        for name, result in results.items():
            analysis = result["analysis"]
            print(f"\\n{name} (λ={result['crowd_aversion']}):")
            print(f"  Equilibrium Type: {analysis['equilibrium_type']}")
            print(f"  Density at Stall: {analysis['density_at_stall']:.3f}")
            print(f"  Maximum Density: {analysis['max_density_value']:.3f}")
            print(f"  Peak Location: x={analysis['max_density_location']:.3f}")
            print(f"  Spatial Spread: {analysis['spatial_spread']:.3f}")

        print(f"\\n Visualization saved to: {plot_path}")

        print("\\n KEY INSIGHTS:")
        print("• Low crowd aversion → Single peak at stall location")
        print("• High crowd aversion → Crater pattern with side peaks")
        print("• Spatial sorting emerges from individual optimization")
        print("• Continuous space prevents total coordination failure")
        print("• Equilibrium depends critically on congestion penalty strength")

        logger.info("Towel Beach Spatial Problem analysis completed successfully")

    except Exception as e:
        logger.error(f"ERROR: Error in spatial beach analysis: {e}")
        raise


if __name__ == "__main__":
    main()
