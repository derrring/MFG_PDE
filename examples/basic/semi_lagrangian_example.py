#!/usr/bin/env python3
"""
Semi-Lagrangian HJB Solver Example

This example demonstrates the semi-Lagrangian method for solving the HJB equation
in Mean Field Games. The semi-Lagrangian approach is particularly effective for:
- Convection-dominated problems
- Large time steps
- Discontinuous solutions
- Monotone solution preservation

The example compares semi-Lagrangian results with standard finite difference methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.factory import create_fast_solver, create_semi_lagrangian_solver
from mfg_pde.utils.integration import trapezoid
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging for research session
configure_research_logging("semi_lagrangian_demo", level="INFO")
logger = get_logger(__name__)


def create_convection_problem():
    """
    Create an MFG problem with strong convection characteristics.

    This problem has a non-trivial transport component that benefits
    from the semi-Lagrangian approach.
    """
    # Define problem parameters
    xmin, xmax = 0.0, 1.0
    Nx = 50  # Moderate spatial resolution
    T = 0.5  # Moderate time horizon
    Nt = 25  # Allows for larger time steps
    sigma = 0.1  # Lower diffusion emphasizes convection
    coefCT = 1.0  # Control cost coefficient

    logger.info(f"Creating convection problem: domain=[{xmin}, {xmax}], Nx={Nx}, T={T}, Nt={Nt}")
    logger.info(f"  σ={sigma} (low diffusion), coefCT={coefCT}")

    return MFGProblem(xmin=xmin, xmax=xmax, Nx=Nx, T=T, Nt=Nt, sigma=sigma, coefCT=coefCT)


def solve_with_methods(problem):
    """
    Solve the MFG problem using different methods for comparison.

    Args:
        problem: MFG problem instance

    Returns:
        Dictionary with results from different methods
    """
    results = {}

    # Method 1: Semi-Lagrangian with linear interpolation
    logger.info("Solving with semi-Lagrangian method (linear interpolation)...")
    try:
        solver_sl_linear = create_semi_lagrangian_solver(
            problem,
            interpolation_method="linear",
            optimization_method="brent",
            characteristic_solver="explicit_euler",
            use_jax=False,  # For compatibility
        )

        result_sl_linear = solver_sl_linear.solve()
        results["semi_lagrangian_linear"] = result_sl_linear

        if hasattr(result_sl_linear, "converged") and result_sl_linear.converged:
            logger.info("  ✓ Semi-Lagrangian (linear) converged successfully")
        else:
            logger.warning("  ⚠ Semi-Lagrangian (linear) did not converge")

    except Exception as e:
        logger.error(f"  ✗ Semi-Lagrangian (linear) failed: {e}")
        results["semi_lagrangian_linear"] = None

    # Method 2: Semi-Lagrangian with cubic interpolation
    logger.info("Solving with semi-Lagrangian method (cubic interpolation)...")
    try:
        solver_sl_cubic = create_semi_lagrangian_solver(
            problem,
            interpolation_method="cubic",
            optimization_method="brent",
            characteristic_solver="explicit_euler",
            use_jax=False,
        )

        result_sl_cubic = solver_sl_cubic.solve()
        results["semi_lagrangian_cubic"] = result_sl_cubic

        if hasattr(result_sl_cubic, "converged") and result_sl_cubic.converged:
            logger.info("  ✓ Semi-Lagrangian (cubic) converged successfully")
        else:
            logger.warning("  ⚠ Semi-Lagrangian (cubic) did not converge")

    except Exception as e:
        logger.error(f"  ✗ Semi-Lagrangian (cubic) failed: {e}")
        results["semi_lagrangian_cubic"] = None

    # Method 3: Standard finite difference for comparison
    logger.info("Solving with standard finite difference method...")
    try:
        from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
        from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver

        hjb_fdm = HJBFDMSolver(problem)
        fp_fdm = FPFDMSolver(problem)

        solver_fdm = create_fast_solver(problem, solver_type="fixed_point", hjb_solver=hjb_fdm, fp_solver=fp_fdm)

        result_fdm = solver_fdm.solve()
        results["finite_difference"] = result_fdm

        if hasattr(result_fdm, "converged") and result_fdm.converged:
            logger.info("  ✓ Finite difference converged successfully")
        else:
            logger.warning("  ⚠ Finite difference did not converge")

    except Exception as e:
        logger.error(f"  ✗ Finite difference failed: {e}")
        results["finite_difference"] = None

    return results


def analyze_results(problem, results):
    """
    Analyze and compare results from different methods.

    Args:
        problem: MFG problem instance
        results: Dictionary with results from different methods
    """
    logger.info("Analyzing results...")

    # Extract successful results
    successful_results = {k: v for k, v in results.items() if v is not None}

    if len(successful_results) == 0:
        logger.error("No methods produced valid results!")
        return

    logger.info(f"Successfully solved with {len(successful_results)} methods:")
    for method_name in successful_results:
        logger.info(f"  - {method_name}")

    # Compare solution properties
    for method_name, result in successful_results.items():
        try:
            if hasattr(result, "U") and hasattr(result, "M"):
                U = result.U
                M = result.M

                # Basic solution statistics
                u_final_norm = np.linalg.norm(U[-1, :])
                x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
                m_mass = trapezoid(M[-1, :], x=x_grid)
                u_variation = np.max(U[-1, :]) - np.min(U[-1, :])

                logger.info(f"  {method_name}:")
                logger.info(f"    Final U norm: {u_final_norm:.4f}")
                logger.info(f"    Final mass: {m_mass:.4f}")
                logger.info(f"    U variation: {u_variation:.4f}")

                # Check for numerical issues
                if np.any(np.isnan(U)) or np.any(np.isinf(U)):
                    logger.warning("    ⚠ Contains NaN/Inf values in U")
                if np.any(np.isnan(M)) or np.any(np.isinf(M)):
                    logger.warning("    ⚠ Contains NaN/Inf values in M")
                if np.any(M < -1e-10):
                    logger.warning("    ⚠ Negative density values detected")

        except Exception as e:
            logger.warning(f"  Error analyzing {method_name}: {e}")


def create_comparison_plots(problem, results):
    """
    Create visualization comparing different methods.

    Args:
        problem: MFG problem instance
        results: Dictionary with results from different methods
    """
    successful_results = {k: v for k, v in results.items() if v is not None}

    if len(successful_results) == 0:
        logger.warning("No successful results to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Semi-Lagrangian vs Finite Difference Methods", fontsize=14)

    x_grid = problem.x if hasattr(problem, "x") else np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)

    # Plot 1: Final value function
    ax1 = axes[0, 0]
    for method_name, result in successful_results.items():
        if hasattr(result, "U"):
            U_final = result.U[-1, :]
            label = method_name.replace("_", " ").title()
            ax1.plot(x_grid, U_final, label=label, linewidth=2)

    ax1.set_xlabel("x")
    ax1.set_ylabel("u(T, x)")
    ax1.set_title("Final Value Function")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final density
    ax2 = axes[0, 1]
    for method_name, result in successful_results.items():
        if hasattr(result, "M"):
            M_final = result.M[-1, :]
            label = method_name.replace("_", " ").title()
            ax2.plot(x_grid, M_final, label=label, linewidth=2)

    ax2.set_xlabel("x")
    ax2.set_ylabel("m(T, x)")
    ax2.set_title("Final Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Value function evolution (for first successful method)
    ax3 = axes[1, 0]
    first_result = next(iter(successful_results.values()))
    if hasattr(first_result, "U"):
        U = first_result.U
        t_grid = np.linspace(0, problem.T, U.shape[0])

        # Plot selected time slices
        time_indices = [0, U.shape[0] // 4, U.shape[0] // 2, 3 * U.shape[0] // 4, -1]
        for i in time_indices:
            label = f"t={t_grid[i]:.2f}"
            ax3.plot(x_grid, U[i, :], label=label, alpha=0.7)

    ax3.set_xlabel("x")
    ax3.set_ylabel("u(t, x)")
    ax3.set_title("Value Function Evolution (First Method)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence comparison (if available)
    ax4 = axes[1, 1]
    for method_name, result in successful_results.items():
        if hasattr(result, "convergence_history"):
            history = result.convergence_history
            iterations = range(len(history))
            label = method_name.replace("_", " ").title()
            ax4.semilogy(iterations, history, label=label, linewidth=2, marker="o", markersize=3)

    ax4.set_xlabel("Picard Iteration")
    ax4.set_ylabel("Residual (log scale)")
    ax4.set_title("Convergence History")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_filename = "semi_lagrangian_comparison.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison plot saved as: {plot_filename}")

    # Show plot if running interactively
    try:
        plt.show()
    except:
        logger.info("Plot display not available (non-interactive environment)")


def demonstrate_characteristics():
    """
    Demonstrate characteristic tracing in semi-Lagrangian method.

    This shows how the method traces characteristics backward in time
    to find departure points for interpolation.
    """
    logger.info("Demonstrating characteristic tracing...")

    # Simple test case
    x_current = 0.5  # Current position
    p_optimal = 0.2  # Optimal control
    dt = 0.02  # Time step

    # Trace characteristic backward (simplified)
    x_departure = x_current - p_optimal * dt

    logger.info(f"  Current position: x = {x_current}")
    logger.info(f"  Optimal control: p = {p_optimal}")
    logger.info(f"  Time step: Δt = {dt}")
    logger.info(f"  Departure point: x_dep = {x_departure}")
    logger.info("  Characteristic traced: x_dep = x - p*Δt")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Semi-Lagrangian HJB Solver Example")
    logger.info("=" * 60)

    try:
        # Create test problem
        problem = create_convection_problem()

        # Demonstrate characteristic tracing
        demonstrate_characteristics()

        # Solve with different methods
        results = solve_with_methods(problem)

        # Analyze results
        analyze_results(problem, results)

        # Create comparison plots
        create_comparison_plots(problem, results)

        logger.info("=" * 60)
        logger.info("Semi-Lagrangian example completed successfully!")

        # Summary
        successful_count = sum(1 for r in results.values() if r is not None)
        logger.info(f"Successfully solved with {successful_count}/{len(results)} methods")

        if "semi_lagrangian_linear" in results and results["semi_lagrangian_linear"] is not None:
            logger.info("✓ Semi-Lagrangian method with linear interpolation: SUCCESS")

        if "semi_lagrangian_cubic" in results and results["semi_lagrangian_cubic"] is not None:
            logger.info("✓ Semi-Lagrangian method with cubic interpolation: SUCCESS")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
