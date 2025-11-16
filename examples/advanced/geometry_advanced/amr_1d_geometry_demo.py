#!/usr/bin/env python3
"""
1D AMR Geometry Demo - Mean Field Game with Adaptive Mesh Refinement
=====================================================================

This example demonstrates the geometry-first API (v0.10.0+) with 1D adaptive
mesh refinement for mean field games. Shows how AMR automatically concentrates
grid points in regions of high solution variation.

Mathematical Problem:
    - Mean field game on [0,1] with adaptive spatial discretization
    - Terminal cost encouraging concentration at x=0.5
    - Running cost penalizing crowding
    - AMR refines mesh near solution features

Key Features:
    - GeometryProtocol-compliant AMR mesh
    - Automatic refinement based on density gradients
    - Comparison with uniform grid
    - Visualization of adaptive grid structure

Created: 2025-11-05
Part of: v0.10.1 AMR GeometryProtocol support
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver
from mfg_pde.geometry.amr.amr_1d import AMRRefinementCriteria, OneDimensionalAMRMesh
from mfg_pde.geometry.domain_1d import Domain1D
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("amr_1d_demo", level="INFO")
logger = get_logger(__name__)


def create_terminal_cost(target_x: float = 0.5, sigma: float = 0.05):
    """
    Create terminal cost encouraging concentration at target location.

    Args:
        target_x: Target location for concentration
        sigma: Width of concentration region

    Returns:
        Terminal cost function φ(x)
    """

    def terminal_cost(x):
        """Terminal cost: φ(x) = (x - target)^2 / (2σ^2)"""
        if x.ndim == 1:
            x = x[:, np.newaxis]
        return 0.5 * ((x - target_x) / sigma) ** 2

    return terminal_cost


def create_running_cost(crowding_penalty: float = 2.0):
    """
    Create running cost penalizing crowding.

    Args:
        crowding_penalty: Penalty coefficient for high density

    Returns:
        Running cost function f(x, m)
    """

    def running_cost(x, m):
        """Running cost: f(x, m) = α * m(x)"""
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if m.ndim == 1:
            m = m[:, np.newaxis]
        return crowding_penalty * m

    return running_cost


def should_refine_cell(cell_data: dict, threshold: float = 0.1) -> bool:
    """
    Refinement criterion based on density gradient.

    Args:
        cell_data: Dictionary with 'density' and 'gradient' arrays
        threshold: Gradient threshold for refinement

    Returns:
        True if cell should be refined
    """
    if "gradient" not in cell_data:
        return False

    gradient = cell_data["gradient"]
    return np.abs(gradient).max() > threshold


def solve_mfg_with_amr(
    initial_num_intervals: int = 20,
    max_refinement: int = 2,
    gradient_threshold: float = 0.5,
    T: float = 1.0,
    Nt: int = 50,
    sigma: float = 0.1,
):
    """
    Solve mean field game with adaptive mesh refinement.

    Args:
        initial_num_intervals: Initial number of intervals
        max_refinement: Maximum AMR refinement level
        gradient_threshold: Threshold for refinement criterion
        T: Terminal time
        Nt: Number of time steps
        sigma: Noise parameter

    Returns:
        Dictionary with solution and mesh information
    """
    logger.info("=" * 70)
    logger.info("Solving MFG with 1D Adaptive Mesh Refinement")
    logger.info("=" * 70)

    # Create base domain
    logger.info("Creating base 1D domain [0,1]")
    from mfg_pde.geometry.boundary.bc_1d import BoundaryConditions

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    # Create refinement criteria
    refinement_criteria = AMRRefinementCriteria(
        max_refinement_levels=max_refinement,
        gradient_threshold=gradient_threshold,
        coarsening_threshold=gradient_threshold * 0.5,
        min_cell_size=0.001,
    )

    # Create AMR mesh
    logger.info(f"Creating 1D AMR mesh (initial intervals: {initial_num_intervals})")
    amr_mesh = OneDimensionalAMRMesh(
        domain_1d=domain, initial_num_intervals=initial_num_intervals, refinement_criteria=refinement_criteria
    )

    logger.info(f"  Initial mesh: {amr_mesh.num_spatial_points} points")
    logger.info(f"  Geometry type: {amr_mesh.geometry_type.value}")

    # Create MFG problem using geometry-first API (v0.10.0+)
    logger.info("\nCreating MFG problem with AMR geometry...")
    problem = ExampleMFGProblem(
        geometry=amr_mesh,  # GeometryProtocol-compliant AMR mesh
        T=T,
        Nt=Nt,
        sigma=sigma,
    )

    # Set terminal and running costs
    problem.gT = create_terminal_cost(target_x=0.5, sigma=0.05)
    problem.f = create_running_cost(crowding_penalty=2.0)

    # Create fast solver
    logger.info("Creating solver...")
    solver = create_fast_solver(problem, method="semi_lagrangian")

    # Solve
    logger.info("\nSolving MFG system...")
    result = solver.solve()

    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Iterations: {result.num_iterations}")
    logger.info(f"  Final error: {result.residual_norm:.2e}")

    # Extract solution on spatial grid
    grid_points = amr_mesh.get_spatial_grid().flatten()
    terminal_value = result.u[:, -1]  # Value function at T
    terminal_density = result.m[:, -1]  # Density at T

    return {
        "mesh": amr_mesh,
        "grid_points": grid_points,
        "u": result.u,
        "m": result.m,
        "terminal_value": terminal_value,
        "terminal_density": terminal_density,
        "result": result,
    }


def solve_mfg_uniform(Nx: int = 100, T: float = 1.0, Nt: int = 50, sigma: float = 0.1):
    """
    Solve mean field game with uniform grid for comparison.

    Args:
        Nx: Number of spatial points
        T: Terminal time
        Nt: Number of time steps
        sigma: Noise parameter

    Returns:
        Dictionary with solution and grid information
    """
    logger.info("\n" + "=" * 70)
    logger.info("Solving MFG with Uniform Grid (comparison)")
    logger.info("=" * 70)

    # Create uniform domain
    logger.info(f"Creating 1D uniform domain ({Nx} points)")
    from mfg_pde.geometry.boundary.bc_1d import BoundaryConditions

    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=BoundaryConditions(type="periodic"))

    # Create MFG problem using geometry-first API
    logger.info("Creating MFG problem with uniform geometry...")
    problem = ExampleMFGProblem(geometry=domain, T=T, Nt=Nt, sigma=sigma)

    # Set same terminal and running costs
    problem.gT = create_terminal_cost(target_x=0.5, sigma=0.05)
    problem.f = create_running_cost(crowding_penalty=2.0)

    # Create fast solver
    logger.info("Creating solver...")
    solver = create_fast_solver(problem, method="semi_lagrangian")

    # Solve
    logger.info("\nSolving MFG system...")
    result = solver.solve()

    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Iterations: {result.num_iterations}")
    logger.info(f"  Final error: {result.residual_norm:.2e}")

    # Extract solution on spatial grid
    grid_points = domain.get_spatial_grid().flatten()
    terminal_value = result.u[:, -1]
    terminal_density = result.m[:, -1]

    return {
        "grid_points": grid_points,
        "u": result.u,
        "m": result.m,
        "terminal_value": terminal_value,
        "terminal_density": terminal_density,
        "result": result,
    }


def visualize_results(amr_solution: dict, uniform_solution: dict):
    """
    Visualize and compare AMR vs uniform grid solutions.

    Args:
        amr_solution: Solution dictionary from solve_mfg_with_amr
        uniform_solution: Solution dictionary from solve_mfg_uniform
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    amr_x = amr_solution["grid_points"]
    amr_u = amr_solution["terminal_value"]
    amr_m = amr_solution["terminal_density"]
    amr_mesh = amr_solution["mesh"]

    uniform_x = uniform_solution["grid_points"]
    uniform_u = uniform_solution["terminal_value"]
    uniform_m = uniform_solution["terminal_density"]

    # Plot 1: AMR mesh structure
    ax = axes[0, 0]
    ax.set_title("1D AMR Mesh Structure", fontsize=12, fontweight="bold")

    # Show cell boundaries with different colors for different levels
    for level in range(amr_mesh.max_level + 1):
        level_cells = [i for i, cell in enumerate(amr_mesh.cells) if cell.level == level and not cell.children]
        if level_cells:
            for i in level_cells:
                cell = amr_mesh.cells[i]
                ax.axvspan(
                    cell.xmin, cell.xmax, alpha=0.3 + 0.1 * level, label=f"Level {level}" if i == level_cells[0] else ""
                )
                ax.axvline(cell.xmin, color="k", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Refinement Level")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Terminal value function comparison
    ax = axes[0, 1]
    ax.set_title("Terminal Value Function u(x,T)", fontsize=12, fontweight="bold")
    ax.plot(uniform_x, uniform_u, "b-", linewidth=2, label="Uniform grid", alpha=0.7)
    ax.plot(amr_x, amr_u, "ro", markersize=6, label="AMR grid", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,T)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Terminal density comparison
    ax = axes[1, 0]
    ax.set_title("Terminal Density m(x,T)", fontsize=12, fontweight="bold")
    ax.plot(uniform_x, uniform_m, "b-", linewidth=2, label="Uniform grid", alpha=0.7)
    ax.plot(amr_x, amr_m, "ro", markersize=6, label="AMR grid", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("m(x,T)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Grid point distribution comparison
    ax = axes[1, 1]
    ax.set_title("Grid Point Distribution", fontsize=12, fontweight="bold")

    # Histogram of grid spacing
    amr_spacing = np.diff(np.sort(amr_x))
    uniform_spacing = np.diff(uniform_x)

    ax.hist(uniform_spacing, bins=20, alpha=0.5, label="Uniform", color="blue")
    ax.hist(amr_spacing, bins=20, alpha=0.5, label="AMR", color="red")
    ax.set_xlabel("Grid Spacing Δx")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print statistics
    print("\n" + "=" * 70)
    print("Solution Statistics")
    print("=" * 70)
    print("AMR mesh:")
    print(f"  Total points: {len(amr_x)}")
    print(f"  Min spacing: {amr_spacing.min():.6f}")
    print(f"  Max spacing: {amr_spacing.max():.6f}")
    print(f"  Spacing ratio: {amr_spacing.max() / amr_spacing.min():.2f}")
    print("\nUniform mesh:")
    print(f"  Total points: {len(uniform_x)}")
    print(f"  Spacing: {uniform_spacing[0]:.6f}")
    print("\nEfficiency:")
    print(f"  AMR uses {len(amr_x) / len(uniform_x) * 100:.1f}% of uniform grid points")

    return fig


def main():
    """Main execution function."""
    logger.info("1D AMR Geometry Demo - v0.10.1 GeometryProtocol")
    logger.info("=" * 70)

    # Problem parameters
    T = 1.0
    Nt = 50
    sigma = 0.1

    # Solve with AMR
    amr_solution = solve_mfg_with_amr(
        initial_num_intervals=20, max_refinement=2, gradient_threshold=0.5, T=T, Nt=Nt, sigma=sigma
    )

    # Solve with uniform grid (using equivalent number of points)
    uniform_solution = solve_mfg_uniform(Nx=amr_solution["mesh"].num_spatial_points, T=T, Nt=Nt, sigma=sigma)

    # Visualize comparison
    logger.info("\n" + "=" * 70)
    logger.info("Creating visualizations...")
    logger.info("=" * 70)

    fig = visualize_results(amr_solution, uniform_solution)

    # Save figure
    import os

    output_dir = "examples/outputs/amr_geometry"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "amr_1d_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("Demo complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
