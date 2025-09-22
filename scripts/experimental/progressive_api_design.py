#!/usr/bin/env python3
"""
Progressive API Design - From Simple to Complex

Shows how users can start simple and gradually access more power
without being overwhelmed by type complexity.
"""

# =============================================================================
# LEVEL 1: "Just make it work" (90% of users)
# =============================================================================


def level_1_example():
    """Dead simple - no types visible to user."""
    from mfg_pde import solve_mfg_problem

    # One function call - everything else is hidden
    result = solve_mfg_problem(problem_type="crowd_dynamics", domain_size=5.0, time_horizon=2.0, crowd_size=1000)

    # Simple access to results
    u_solution = result.u  # Value function
    m_solution = result.m  # Density
    converged = result.success

    return result


# =============================================================================
# LEVEL 2: "I want some control" (8% of users)
# =============================================================================


def level_2_example():
    """Expose key concepts but keep types simple."""
    from mfg_pde import create_problem, create_solver

    # Two-step process with some customization
    problem = create_problem(
        type="crowd_dynamics",
        domain=(0, 5),
        hamiltonian="quadratic",  # String-based config
        initial_density="gaussian",
    )

    solver = create_solver(
        problem,
        method="fdm",  # Finite difference method
        grid_size=100,
        max_iterations=200,
    )

    result = solver.solve()
    return result


# =============================================================================
# LEVEL 3: "I know what I'm doing" (2% of users)
# =============================================================================


def level_3_example():
    """Full access to internal types and customization."""
    import numpy as np

    from mfg_pde.alg.hjb_solvers import HJBSemiLagrangianSolver
    from mfg_pde.config import HJBSolverConfig
    from mfg_pde.core import LagrangianMFGProblem, MFGComponents

    # Custom Hamiltonian
    def custom_hamiltonian(x, p, m, t):
        return 0.5 * p**2 + x * m  # Custom coupling

    # Full type-safe configuration
    components = MFGComponents(
        hamiltonian_func=custom_hamiltonian,
        initial_density_func=lambda x: np.exp(-(x**2)),
        final_value_func=lambda x: x**2,
    )

    problem = LagrangianMFGProblem(xmin=0.0, xmax=1.0, Nx=200, T=1.0, Nt=100, sigma=0.1, components=components)

    config = HJBSolverConfig(
        interpolation_method="cubic", optimization_method="brent", characteristic_solver="runge_kutta"
    )

    solver = HJBSemiLagrangianSolver(problem, config=config)
    result = solver.solve(max_iterations=500, tolerance=1e-8)

    return result


# =============================================================================
# KEY INSIGHT: Same underlying system, different interfaces
# =============================================================================

"""
All three levels use the SAME internal implementation:
- Level 1: solve_mfg_problem() internally calls Level 2 functions
- Level 2: create_problem() internally creates Level 3 objects
- Level 3: Direct access to full type system

Benefits:
1. Beginners aren't overwhelmed
2. Experts get full power
3. Gradual learning curve
4. Single codebase to maintain
"""
