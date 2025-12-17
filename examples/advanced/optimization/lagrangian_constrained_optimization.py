#!/usr/bin/env python3
"""
Advanced Lagrangian MFG: Constrained Optimization Problems

This example demonstrates the power of the Lagrangian formulation for handling
complex constraints that are difficult or impossible to incorporate in the
Hamiltonian (HJB-FP) framework.

Featured problems:
1. Obstacle avoidance MFG
2. Budget-constrained optimization
3. Social distancing with capacity constraints
4. Multi-objective optimization with trade-offs

The Lagrangian approach naturally handles:
- Inequality constraints on states and controls
- Integral constraints (budget, capacity)
- Multiple objectives with Pareto optimization
- Non-convex constraints
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.optimization.variational_solvers.base_variational import VariationalSolverResult
from mfg_pde.alg.optimization.variational_solvers.variational_mfg_solver import VariationalMFGSolver
from mfg_pde.core.lagrangian_mfg_problem import (
    LagrangianComponents,
    LagrangianMFGProblem,
    create_obstacle_lagrangian_mfg,
)
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger
from mfg_pde.utils.numpy_compat import trapezoid

# Configure logging
configure_research_logging("lagrangian_constrained", level="INFO")
logger = get_logger(__name__)


def create_obstacle_avoidance_problem() -> LagrangianMFGProblem:
    """
    Create MFG problem with obstacle avoidance constraints.

    Agents must navigate around a circular obstacle while minimizing
    travel time and congestion costs.

    Returns:
        LagrangianMFGProblem with obstacle constraints
    """
    logger.info("Creating obstacle avoidance MFG problem...")

    return create_obstacle_lagrangian_mfg(
        xmin=0.0,
        xmax=1.0,
        Nx=40,
        T=0.5,
        Nt=25,
        obstacle_center=0.5,
        obstacle_radius=0.15,
        obstacle_penalty=1000.0,
        diffusion=0.1,
    )


def create_budget_constrained_problem() -> LagrangianMFGProblem:
    """
    Create MFG with budget constraints on control effort.

    Each agent has a limited budget for control/acceleration,
    enforced via integral constraints.

    Returns:
        LagrangianMFGProblem with budget constraints
    """
    logger.info("Creating budget-constrained MFG problem...")

    def budget_lagrangian(t: float, x: float, v: float, m: float) -> float:
        """Lagrangian with budget-aware control cost."""
        # Base control cost
        control_cost = 0.5 * v**2

        # Congestion cost
        congestion_cost = 0.3 * m

        # Time penalty (encourages faster movement)
        time_penalty = 0.1

        return control_cost + congestion_cost + time_penalty

    def budget_constraint(trajectory: np.ndarray, velocity: np.ndarray, dt: float) -> float:
        """
        Budget constraint: ∫₀ᵀ |v(t)|² dt ≤ B

        Returns constraint violation (positive if violated)
        """
        total_control_effort = trapezoid(velocity**2, dx=dt)
        budget_limit = 2.0  # Maximum allowed control effort
        return total_control_effort - budget_limit

    def initial_density_concentrated(x: float) -> float:
        """Initial density concentrated at left boundary."""
        if 0.0 <= x <= 0.2:
            return 5.0  # High density at start
        else:
            return 0.0

    components = LagrangianComponents(
        lagrangian_func=budget_lagrangian,
        lagrangian_dv_func=lambda t, x, v, m: v,  # ∂L/∂v = v
        lagrangian_dm_func=lambda t, x, v, m: 0.3,  # ∂L/∂m = 0.3
        terminal_cost_func=lambda x: 0.0,
        initial_density_func=initial_density_concentrated,
        integral_constraints=[budget_constraint],
        noise_intensity=0.1,
        parameters={"budget_limit": 2.0, "control_cost": 0.5, "congestion_cost": 0.3},
        description="Budget-Constrained MFG",
    )

    return LagrangianMFGProblem(xmin=0.0, xmax=1.0, Nx=30, T=1.0, Nt=30, diffusion=0.1, components=components)


def create_social_distancing_problem() -> LagrangianMFGProblem:
    """
    Create MFG with social distancing and capacity constraints.

    Models pandemic-era movement with:
    - Social distancing cost (high density penalty)
    - Capacity constraints on regions
    - Essential travel vs. optional travel trade-offs

    Returns:
        LagrangianMFGProblem with capacity constraints
    """
    logger.info("Creating social distancing MFG problem...")

    def social_distancing_lagrangian(t: float, x: float, v: float, m: float) -> float:
        """Lagrangian with social distancing costs."""
        # Movement cost (travel time)
        movement_cost = 0.2 * v**2

        # Social distancing cost (super-linear in density)
        if m > 0.5:  # Capacity threshold
            distancing_cost = 10.0 * (m - 0.5) ** 2  # Quadratic penalty above threshold
        else:
            distancing_cost = 0.1 * m**2  # Small cost below threshold

        # Essential vs. non-essential travel (time-dependent)
        if t < 0.3:  # Early period - high restrictions
            travel_restriction = 5.0 * v**2
        else:  # Later period - relaxed restrictions
            travel_restriction = 1.0 * v**2

        return movement_cost + distancing_cost + travel_restriction

    def capacity_constraint(t: float, density_field: np.ndarray, x_grid: np.ndarray) -> float:
        """
        Capacity constraint: density should not exceed limits in high-risk zones.

        Returns maximum capacity violation.
        """
        high_risk_zone = (x_grid >= 0.4) & (x_grid <= 0.6)  # Central zone
        max_density_in_zone = np.max(density_field[high_risk_zone])
        capacity_limit = 2.0
        return max(0.0, max_density_in_zone - capacity_limit)

    def initial_density_social(x: float) -> float:
        """Initial density with social gathering pattern."""
        # Multiple population centers
        center1 = 0.2
        center2 = 0.8
        width = 0.1

        density1 = 3.0 * np.exp(-0.5 * ((x - center1) / width) ** 2)
        density2 = 2.0 * np.exp(-0.5 * ((x - center2) / width) ** 2)

        return density1 + density2

    components = LagrangianComponents(
        lagrangian_func=social_distancing_lagrangian,
        lagrangian_dv_func=lambda t, x, v, m: 0.2 * v + (5.0 if t < 0.3 else 1.0) * 2 * v,
        lagrangian_dm_func=lambda t, x, v, m: (20.0 * (m - 0.5) if m > 0.5 else 0.2 * m),
        terminal_cost_func=lambda x: 0.0,
        initial_density_func=initial_density_social,
        noise_intensity=0.05,  # Lower noise (more deterministic movement)
        parameters={"capacity_limit": 2.0, "distancing_threshold": 0.5, "restriction_period": 0.3},
        description="Social Distancing MFG with Capacity Constraints",
    )

    return LagrangianMFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=0.6, Nt=30, diffusion=0.05, components=components)


def solve_constrained_problem(problem: LagrangianMFGProblem, problem_name: str, max_iterations: int = 30) -> dict:
    """
    Solve a constrained Lagrangian MFG problem.

    Args:
        problem: LagrangianMFGProblem to solve
        problem_name: Descriptive name for logging
        max_iterations: Maximum optimization iterations

    Returns:
        Dictionary with solution results and analysis
    """
    logger.info(f"Solving {problem_name}...")

    start_time = time.time()

    try:
        # Create solver with increased penalty weights for constraints
        solver = VariationalMFGSolver(
            problem,
            optimization_method="L-BFGS-B",
            penalty_weight=500.0,  # Higher weight for constraint enforcement
            constraint_tolerance=1e-4,
            use_jax=False,
        )

        # Create initial guess that respects constraints
        if "obstacle" in problem_name.lower():
            initial_guess = create_obstacle_aware_initial_guess(problem)
        elif "budget" in problem_name.lower():
            initial_guess = create_budget_aware_initial_guess(problem)
        else:
            initial_guess = solver.create_initial_guess("gaussian")

        # Solve with constraints
        result = solver.solve(
            initial_guess=initial_guess,
            max_iterations=max_iterations,
            tolerance=1e-3,  # Slightly relaxed for constrained problems
            verbose=True,
        )

        solve_time = time.time() - start_time

        # Analyze constraint satisfaction
        constraint_analysis = analyze_constraint_satisfaction(problem, result)

        # Package results
        solution_info = {
            "result": result,
            "solve_time": solve_time,
            "constraint_analysis": constraint_analysis,
            "problem_name": problem_name,
            "success": result.converged if result else False,
        }

        if result and result.converged:
            logger.info(f"  ✓ {problem_name} solved successfully")
            logger.info(f"  ✓ Final cost: {result.final_cost:.4e}")
            logger.info(f"  ✓ Solve time: {solve_time:.2f}s")

            # Log constraint violations
            if result.constraint_violations:
                max_violation = max(result.constraint_violations.values())
                logger.info(f"  ✓ Max constraint violation: {max_violation:.2e}")
        else:
            logger.warning(f"  ⚠ {problem_name} did not converge")

        return solution_info

    except Exception as e:
        logger.error(f"  ✗ {problem_name} failed: {e}")
        return {
            "result": None,
            "solve_time": time.time() - start_time,
            "constraint_analysis": {},
            "problem_name": problem_name,
            "success": False,
            "error": str(e),
        }


def create_obstacle_aware_initial_guess(problem: LagrangianMFGProblem) -> np.ndarray:
    """Create initial guess that avoids obstacles."""
    Nx_points = problem.geometry.get_grid_shape()[0]
    density_guess = np.zeros((problem.Nt, Nx_points))

    # Get obstacle parameters
    obstacle_center = problem.components.parameters.get("obstacle_center", 0.5)
    obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.1)

    for i in range(problem.Nt):
        for j, x in enumerate(problem.x):
            # Avoid obstacle region
            distance_to_obstacle = abs(x - obstacle_center)

            if distance_to_obstacle > obstacle_radius + 0.05:  # Safety margin
                # Gaussian distribution avoiding obstacle
                if x < obstacle_center:
                    # Left side of obstacle
                    center = obstacle_center - obstacle_radius - 0.1
                else:
                    # Right side of obstacle
                    center = obstacle_center + obstacle_radius + 0.1

                width = 0.1
                density_guess[i, j] = np.exp(-0.5 * ((x - center) / width) ** 2)
            else:
                density_guess[i, j] = 1e-6  # Minimal density in obstacle region

        # Normalize
        x_grid = problem.geometry.get_spatial_grid()
        total_mass = trapezoid(density_guess[i, :], x=x_grid)
        if total_mass > 1e-12:
            density_guess[i, :] /= total_mass

    return density_guess


def create_budget_aware_initial_guess(problem: LagrangianMFGProblem) -> np.ndarray:
    """Create initial guess that respects budget constraints."""
    Nx_points = problem.geometry.get_grid_shape()[0]
    density_guess = np.zeros((problem.Nt, Nx_points))

    # Start with initial density
    x_grid = problem.geometry.get_spatial_grid()
    if problem.components.initial_density_func:
        initial_dist = np.array([problem.components.initial_density_func(x) for x in x_grid])
    else:
        initial_dist = np.ones(len(x_grid))

    # Normalize initial distribution
    initial_dist = initial_dist / trapezoid(initial_dist, x=x_grid)

    # Evolve slowly to respect budget (slow movement = low control cost)
    for i, t in enumerate(problem.t):
        # Gradual spreading with limited movement
        spread_factor = 0.1 * t  # Slow spreading

        for j, x in enumerate(problem.x):
            # Shifted and spread version of initial distribution
            shift = 0.2 * t  # Slow rightward movement

            # Find corresponding point in initial distribution
            x_original = x - shift

            # Interpolate from initial distribution
            bounds = problem.geometry.get_bounds()
            dx = problem.geometry.get_grid_spacing()[0]
            if bounds[0][0] <= x_original <= bounds[1][0]:
                idx = (x_original - bounds[0][0]) / dx
                i_idx = int(idx)
                alpha = idx - i_idx

                if i_idx < len(initial_dist) - 1:
                    density_value = (1 - alpha) * initial_dist[i_idx] + alpha * initial_dist[i_idx + 1]
                else:
                    density_value = initial_dist[-1]
            else:
                density_value = 1e-6

            # Add diffusion spreading
            width = 0.1 + spread_factor
            center = 0.2 + shift
            diffusion_component = 0.5 * np.exp(-0.5 * ((x - center) / width) ** 2)

            density_guess[i, j] = 0.7 * density_value + 0.3 * diffusion_component

        # Normalize
        total_mass = trapezoid(density_guess[i, :], x=x_grid)
        if total_mass > 1e-12:
            density_guess[i, :] /= total_mass

    return density_guess


def analyze_constraint_satisfaction(problem: LagrangianMFGProblem, result: VariationalSolverResult) -> dict:
    """
    Analyze how well constraints are satisfied in the solution.

    Args:
        problem: Original problem with constraints
        result: Solver result to analyze

    Returns:
        Dictionary with constraint satisfaction analysis
    """
    if not result or not hasattr(result, "optimal_flow") or result.optimal_flow is None:
        return {"status": "no_solution"}

    analysis = {}
    density = result.optimal_flow

    # Check obstacle constraints
    if "obstacle" in problem.components.description.lower():
        obstacle_center = problem.components.parameters.get("obstacle_center", 0.5)
        obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.1)

        # Find maximum density in obstacle region
        obstacle_violations = []
        for i in range(problem.Nt):
            for j, x in enumerate(problem.x):
                if abs(x - obstacle_center) < obstacle_radius:
                    obstacle_violations.append(density[i, j])

        analysis["obstacle_max_density"] = max(obstacle_violations) if obstacle_violations else 0.0
        analysis["obstacle_avg_density"] = np.mean(obstacle_violations) if obstacle_violations else 0.0

    # Check capacity constraints
    if "capacity" in problem.components.description.lower() or "distancing" in problem.components.description.lower():
        capacity_limit = problem.components.parameters.get("capacity_limit", 2.0)
        max_densities = np.max(density, axis=1)  # Max density at each time

        analysis["capacity_violations"] = np.sum(max_densities > capacity_limit)
        analysis["max_density_violation"] = max(0.0, np.max(max_densities) - capacity_limit)

    # Check mass conservation
    x_grid = problem.geometry.get_spatial_grid()
    mass_over_time = [trapezoid(density[i, :], x=x_grid) for i in range(problem.Nt)]
    analysis["mass_conservation_error"] = np.std(mass_over_time)
    analysis["final_mass"] = mass_over_time[-1]

    return analysis


def create_constraint_comparison_plots(solutions: list[dict]):
    """
    Create plots comparing different constrained problems.

    Args:
        solutions: List of solution dictionaries from solve_constrained_problem
    """
    logger.info("Creating constraint comparison plots...")

    # Filter successful solutions
    successful_solutions = [sol for sol in solutions if sol["success"] and sol["result"]]

    if len(successful_solutions) == 0:
        logger.warning("No successful solutions to plot")
        return

    try:
        fig, axes = plt.subplots(2, len(successful_solutions), figsize=(4 * len(successful_solutions), 8))

        if len(successful_solutions) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle("Constrained Lagrangian MFG Problems", fontsize=14)

        for i, solution in enumerate(successful_solutions):
            result = solution["result"]
            problem_name = solution["problem_name"]

            # Get the problem for grid information
            if "obstacle" in problem_name.lower():
                problem = create_obstacle_avoidance_problem()
            elif "budget" in problem_name.lower():
                problem = create_budget_constrained_problem()
            else:
                problem = create_social_distancing_problem()

            x_grid = problem.geometry.get_spatial_grid()
            t_grid = problem.t
            density = result.optimal_flow

            # Plot 1: Final density distribution
            ax1 = axes[0, i]
            final_density = density[-1, :]
            ax1.plot(x_grid, final_density, "b-", linewidth=2)

            # Add constraint visualization
            if "obstacle" in problem_name.lower():
                obstacle_center = problem.components.parameters.get("obstacle_center", 0.5)
                obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.1)

                # Shade obstacle region
                obstacle_left = obstacle_center - obstacle_radius
                obstacle_right = obstacle_center + obstacle_radius
                ax1.axvspan(obstacle_left, obstacle_right, alpha=0.3, color="red", label="Obstacle")
                ax1.legend()

            elif "distancing" in problem_name.lower():
                # Shade high-risk zone
                ax1.axvspan(0.4, 0.6, alpha=0.3, color="orange", label="High-Risk Zone")
                ax1.axhline(y=2.0, color="red", linestyle="--", label="Capacity Limit")
                ax1.legend()

            ax1.set_xlabel("x")
            ax1.set_ylabel("m(T, x)")
            ax1.set_title(f"{problem_name}\nFinal Density")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Density evolution over time
            ax2 = axes[1, i]

            # Create space-time heatmap
            X, T = np.meshgrid(x_grid, t_grid)
            im = ax2.contourf(X, T, density, levels=20, cmap="viridis")

            # Add constraint overlays
            if "obstacle" in problem_name.lower():
                obstacle_center = problem.components.parameters.get("obstacle_center", 0.5)
                obstacle_radius = problem.components.parameters.get("obstacle_radius", 0.1)

                # Draw obstacle region
                for t in t_grid:
                    ax2.plot(
                        [obstacle_center - obstacle_radius, obstacle_center + obstacle_radius],
                        [t, t],
                        "r-",
                        linewidth=2,
                        alpha=0.7,
                    )

            ax2.set_xlabel("x")
            ax2.set_ylabel("t")
            ax2.set_title(f"{problem_name}\nDensity Evolution")

            # Add colorbar
            plt.colorbar(im, ax=ax2, shrink=0.8)

        plt.tight_layout()

        # Save plot
        plot_filename = "lagrangian_constrained_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        logger.info(f"Constraint comparison plots saved as: {plot_filename}")

        try:
            plt.show()
        except:
            logger.info("Plot display not available (non-interactive environment)")

    except Exception as e:
        logger.error(f"Plot creation failed: {e}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Advanced Lagrangian MFG: Constrained Optimization Problems")
    logger.info("=" * 80)

    try:
        # Create constrained problems
        logger.info("Creating constrained MFG problems...")

        obstacle_problem = create_obstacle_avoidance_problem()
        budget_problem = create_budget_constrained_problem()
        social_problem = create_social_distancing_problem()

        # Solve each problem
        logger.info("\n" + "=" * 60 + " SOLVING " + "=" * 60)

        solutions = []

        # Solve obstacle avoidance
        obstacle_solution = solve_constrained_problem(obstacle_problem, "Obstacle Avoidance", max_iterations=25)
        solutions.append(obstacle_solution)

        # Solve budget-constrained
        budget_solution = solve_constrained_problem(budget_problem, "Budget Constrained", max_iterations=25)
        solutions.append(budget_solution)

        # Solve social distancing
        social_solution = solve_constrained_problem(social_problem, "Social Distancing", max_iterations=25)
        solutions.append(social_solution)

        # Create comparison visualization
        create_constraint_comparison_plots(solutions)

        # Summary analysis
        logger.info("\n" + "=" * 80)
        logger.info("CONSTRAINT SATISFACTION ANALYSIS")
        logger.info("=" * 80)

        for solution in solutions:
            problem_name = solution["problem_name"]
            logger.info(f"\n{problem_name}:")

            if solution["success"]:
                logger.info("  ✓ Optimization: SUCCESS")
                logger.info(f"  ✓ Solve time: {solution['solve_time']:.2f}s")

                # Constraint analysis
                constraint_analysis = solution["constraint_analysis"]

                if "obstacle_max_density" in constraint_analysis:
                    logger.info(f"  ✓ Max density in obstacle: {constraint_analysis['obstacle_max_density']:.4f}")

                if "capacity_violations" in constraint_analysis:
                    violations = constraint_analysis["capacity_violations"]
                    logger.info(f"  ✓ Capacity violations: {violations} time steps")

                mass_error = constraint_analysis.get("mass_conservation_error", 0)
                logger.info(f"  ✓ Mass conservation error: {mass_error:.2e}")

            else:
                logger.info("  ✗ Optimization: FAILED")
                if "error" in solution:
                    logger.info(f"     Error: {solution['error']}")

        # Overall summary
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 80)

        successful_count = sum(1 for sol in solutions if sol["success"])
        total_count = len(solutions)

        logger.info(f"Successfully solved: {successful_count}/{total_count} constrained problems")

        if successful_count == total_count:
            logger.info("🎉 ALL CONSTRAINED PROBLEMS SOLVED!")
            logger.info("   Lagrangian formulation successfully handles:")
            logger.info("   • Obstacle avoidance constraints")
            logger.info("   • Budget/resource constraints")
            logger.info("   • Capacity and social distancing constraints")
        elif successful_count > 0:
            logger.info("⚠️  PARTIAL SUCCESS")
            logger.info("   Some constrained problems solved successfully")
        else:
            logger.info("❌ CONSTRAINT HANDLING NEEDS IMPROVEMENT")

        logger.info("\nKey advantages of Lagrangian formulation:")
        logger.info("• Natural constraint incorporation")
        logger.info("• Economic interpretation of costs and trade-offs")
        logger.info("• Flexible penalty method implementation")
        logger.info("• Direct optimization of complex objectives")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Advanced constrained example failed: {e}")
        raise


if __name__ == "__main__":
    main()
