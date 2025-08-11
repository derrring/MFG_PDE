#!/usr/bin/env python3
"""
Basic AMR (Adaptive Mesh Refinement) Example for MFG_PDE

This example demonstrates how to use the Adaptive Mesh Refinement solver
for Mean Field Games problems, showcasing automatic mesh adaptation
based on solution error estimates.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver


def main():
    """Demonstrate AMR solver capabilities."""

    print("Creating AMR MFG Problem...")

    # Create a problem with sharp features that benefit from AMR
    problem = ExampleMFGProblem(Nx=64, Nt=50, xmin=-2.0, xmax=2.0, T=1.0, sigma=0.1, coefCT=0.5)

    print("Creating AMR solver...")

    # Create AMR solver with adaptive mesh refinement
    amr_solver = create_amr_solver(
        problem,
        error_threshold=1e-4,  # Refine when error exceeds this threshold
        max_levels=4,  # Maximum 4 levels of refinement
        base_solver="fixed_point",
        amr_frequency=5,  # Adapt mesh every 5 iterations
        max_amr_cycles=3,  # Maximum 3 AMR cycles
        backend="auto",  # Automatic backend selection
    )

    print("Solving with AMR...")

    # Solve the MFG problem with automatic mesh adaptation
    result = amr_solver.solve(max_iterations=100, tolerance=1e-6, verbose=True)

    print(f"AMR Solution completed!")
    print(f"- Convergence: {'Yes' if result.convergence_achieved else 'No'}")
    print(f"- Final error: {result.final_error:.2e}")
    print(f"- Execution time: {result.execution_time:.2f}s")

    # Display AMR statistics
    if 'amr_stats' in result.solver_info:
        amr_stats = result.solver_info['amr_stats']
        print(f"\nAMR Statistics:")
        print(f"- Total refinements: {amr_stats['total_refinements']}")
        print(f"- Total coarsenings: {amr_stats['total_coarsenings']}")
        print(f"- Adaptation cycles: {amr_stats['adaptation_cycles']}")

    if 'final_mesh_stats' in result.solver_info:
        mesh_stats = result.solver_info['final_mesh_stats']
        print(f"\nFinal Mesh Statistics:")
        print(f"- Total cells: {mesh_stats['total_cells']}")
        print(f"- Leaf cells: {mesh_stats['leaf_cells']}")
        print(f"- Maximum level: {mesh_stats['max_level']}")
        print(f"- Refinement ratio: {mesh_stats['refinement_ratio']:.2f}")

    # Plot results
    create_amr_plots(result, problem)

    print("AMR example completed successfully!")


def create_amr_plots(result, problem):
    """Create visualization plots for AMR results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('AMR MFG Solver Results', fontsize=16)

    # Solution arrays
    U = result.U
    M = result.M

    # Create spatial grid for plotting
    x = np.linspace(problem.xmin, problem.xmax, U.shape[0])
    X, Y = np.meshgrid(x, x)

    # Plot value function U
    im1 = axes[0, 0].contourf(X, Y, U.T, levels=20, cmap='viridis')
    axes[0, 0].set_title('Value Function U(x)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot density function M
    im2 = axes[0, 1].contourf(X, Y, M.T, levels=20, cmap='plasma')
    axes[0, 1].set_title('Density Function M(x)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot convergence history
    if result.convergence_history:
        axes[1, 0].semilogy(result.convergence_history)
        axes[1, 0].set_title('Convergence History')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].grid(True)

    # Plot mesh efficiency (if available)
    if 'amr_stats' in result.solver_info:
        amr_stats = result.solver_info['amr_stats']
        if amr_stats['mesh_efficiency']:
            axes[1, 1].plot(amr_stats['mesh_efficiency'], 'o-')
            axes[1, 1].set_title('Mesh Efficiency Evolution')
            axes[1, 1].set_xlabel('AMR Cycle')
            axes[1, 1].set_ylabel('Cells per Unit Area')
            axes[1, 1].grid(True)

    plt.tight_layout()

    # Try to save the plot
    try:
        plt.savefig('amr_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'amr_results.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Show plot if possible
    try:
        plt.show()
    except Exception:
        print("Display not available, plot saved to file.")


if __name__ == "__main__":
    main()
