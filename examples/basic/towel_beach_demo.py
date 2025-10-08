"""
Towel on the Beach Problem: Spatial Competition MFG.

This example demonstrates spatial competition where agents choose beach positions
balancing proximity to an ice cream stall against crowding avoidance.

Mathematical Formulation:
    State: x ∈ [0,1] (position on beach)
    Dynamics: dx_t = u_t dt + σ dW_t

    Running cost:
        L(x, u, m) = |x - x_stall| + λ log(m(x)) + (1/2)|u|²
                     [proximity]      [congestion]   [movement]

    HJB equation (backward):
        -∂u/∂t + (1/2)|∇u|² - |x - x_stall| - λ log(m) - (σ²/2)Δu = 0

    FPK equation (forward):
        ∂m/∂t + ∇·(m ∇u) - (σ²/2)Δm = 0

    Equilibrium: Find (u, m) satisfying both equations.

Phase Transitions:
    - λ < λ_c1: Single peak at stall (proximity dominates)
    - λ_c1 < λ < λ_c2: Mixed pattern
    - λ > λ_c2: Crater equilibrium (congestion avoidance dominates)

References:
    - docs/theory/spatial_competition_mfg.md: Comprehensive formulation
    - Hotelling (1929): Stability in competition
    - Lasry & Lions (2007): Mean field games

Example Usage:
    python examples/basic/towel_beach_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("towel_beach_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_towel_beach_problem(
    stall_position: float = 0.5,
    crowd_aversion: float = 1.0,
    xmin: float = 0.0,
    xmax: float = 1.0,
    Nx: int = 101,
    T: float = 1.0,
    Nt: int = 101,
    sigma: float = 0.1,
):
    """
    Create Towel on Beach MFG problem.

    Args:
        stall_position: Ice cream stall location x_stall ∈ [0,1]
        crowd_aversion: Congestion parameter λ > 0
        xmin: Beach left boundary
        xmax: Beach right boundary
        Nx: Number of spatial grid points
        T: Time horizon
        Nt: Number of time steps
        sigma: Diffusion coefficient

    Returns:
        ExampleMFGProblem configured for spatial competition
    """

    def hamiltonian(x, p, m):
        """
        H(x, p, m) = (1/2)|p|² - |x - x_stall| - λ log(m)

        Args:
            x: Position
            p: Momentum (∇u)
            m: Density

        Returns:
            Hamiltonian value
        """
        # Kinetic term (optimal control u* = -p)
        kinetic = 0.5 * p**2

        # Proximity cost (distance to stall)
        proximity = -np.abs(x - stall_position)

        # Congestion cost (logarithmic penalty)
        # Add small regularization to prevent log(0)
        m_reg = np.maximum(m, 1e-10)
        congestion = -crowd_aversion * np.log(m_reg)

        return kinetic + proximity + congestion

    # Create problem using ExampleMFGProblem
    problem = ExampleMFGProblem(
        xmin=xmin,
        xmax=xmax,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        hamiltonian=hamiltonian,
    )

    # Set initial distribution: Uniform on beach
    x_grid = problem.x
    m0 = np.ones_like(x_grid) / (xmax - xmin)  # Uniform density
    problem.set_initial_distribution(m0)

    return problem


def plot_results(x_grid, m_final, u_final, stall_position, crowd_aversion):
    """
    Visualize spatial equilibrium.

    Args:
        x_grid: Spatial grid
        m_final: Equilibrium density m(x)
        u_final: Equilibrium value function u(x)
        stall_position: Stall location
        crowd_aversion: λ parameter
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Equilibrium density
    ax = axes[0]
    ax.plot(x_grid, m_final, "b-", linewidth=2.5, label="Equilibrium density m(x)")
    ax.axvline(stall_position, color="red", linestyle="--", linewidth=2, label=f"Ice cream stall (x={stall_position})")
    ax.fill_between(x_grid, 0, m_final, alpha=0.3, color="skyblue")
    ax.set_xlabel("Beach position x", fontsize=13)
    ax.set_ylabel("Density m(x)", fontsize=13)
    ax.set_title(f"Spatial Equilibrium (λ = {crowd_aversion})", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_grid[0], x_grid[-1]])

    # Annotate equilibrium pattern
    max_density_pos = x_grid[np.argmax(m_final)]
    pattern_type = "Single Peak" if np.abs(max_density_pos - stall_position) < 0.1 else "Crater/Mixed"
    ax.text(
        0.02,
        0.95,
        f"Pattern: {pattern_type}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Panel 2: Value function
    ax = axes[1]
    ax.plot(x_grid, u_final, "g-", linewidth=2.5, label="Value function u(x)")
    ax.axvline(stall_position, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Beach position x", fontsize=13)
    ax.set_ylabel("Value u(x)", fontsize=13)
    ax.set_title("Optimal Value Function", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_grid[0], x_grid[-1]])

    plt.tight_layout()
    return fig


def main():
    """
    Solve Towel Beach problem for different crowd aversion levels.
    """
    logger.info("=" * 70)
    logger.info("Towel on the Beach: Spatial Competition Mean Field Game")
    logger.info("=" * 70)

    # Parameters
    stall_pos = 0.5  # Center of beach
    lambda_values = [0.5, 1.0, 2.0]  # Different crowd aversion levels

    for lambda_crowd in lambda_values:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Solving for λ = {lambda_crowd} (Crowd Aversion)")
        logger.info(f"{'=' * 70}")

        # Create problem
        logger.info("\n[1/3] Creating spatial competition problem...")
        problem = create_towel_beach_problem(
            stall_position=stall_pos,
            crowd_aversion=lambda_crowd,
            xmin=0.0,
            xmax=1.0,
            Nx=101,
            T=1.0,
            Nt=101,
            sigma=0.1,
        )

        logger.info(f"  Beach domain: [{problem.xmin}, {problem.xmax}]")
        logger.info(f"  Stall position: x = {stall_pos}")
        logger.info(f"  Crowd aversion: λ = {lambda_crowd}")
        logger.info(f"  Grid: Nx = {problem.Nx}")

        # Solve using factory (fast solver)
        logger.info("\n[2/3] Solving MFG system...")
        solver = create_fast_solver(problem)

        result = solver.solve()

        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  Iterations: {result.num_iterations}")
        logger.info(f"  Final error: {result.final_error:.2e}")

        # Extract equilibrium (final time)
        x_grid = problem.x
        m_final = result.m_traj[-1, :]  # Equilibrium density
        u_final = result.u_traj[-1, :]  # Equilibrium value

        # Analysis
        logger.info("\n[3/3] Equilibrium Analysis:")
        max_density = np.max(m_final)
        max_density_pos = x_grid[np.argmax(m_final)]
        mean_position = np.sum(x_grid * m_final) / np.sum(m_final)

        logger.info(f"  Max density: {max_density:.3f} at x = {max_density_pos:.3f}")
        logger.info(f"  Mean position: {mean_position:.3f}")
        logger.info(f"  Distance from stall: {np.abs(max_density_pos - stall_pos):.3f}")

        # Determine pattern
        if np.abs(max_density_pos - stall_pos) < 0.1:
            pattern = "SINGLE PEAK (proximity dominates)"
        elif max_density_pos < stall_pos - 0.1 or max_density_pos > stall_pos + 0.1:
            pattern = "CRATER/MIXED (congestion avoidance strong)"
        else:
            pattern = "MIXED PATTERN"
        logger.info(f"  Pattern: {pattern}")

        # Visualize
        logger.info("\nGenerating visualization...")
        fig = plot_results(x_grid, m_final, u_final, stall_pos, lambda_crowd)

        output_path = OUTPUT_DIR / f"towel_beach_lambda_{lambda_crowd:.1f}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")

        plt.close(fig)

    logger.info("\n" + "=" * 70)
    logger.info("Towel Beach Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Insights:")
    logger.info("  - Low λ: Agents cluster near stall (proximity benefit)")
    logger.info("  - High λ: Agents avoid stall area (congestion cost)")
    logger.info("  - Phase transition controlled by single parameter λ")
    logger.info("\nCompare generated figures to see equilibrium evolution!")


if __name__ == "__main__":
    main()
