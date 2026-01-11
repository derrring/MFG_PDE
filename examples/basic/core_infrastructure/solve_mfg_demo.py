#!/usr/bin/env python3
"""
MFG Problem Solving Demonstration

Demonstrates the primary API for solving Mean Field Games problems.

Primary API: problem.solve()
----------------------------
The recommended way to solve MFG problems is using the solve() method
directly on MFGProblem instances.

Run:
    python examples/basic/core_infrastructure/solve_mfg_demo.py
"""

from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid


def demo_simple_usage():
    """Simplest usage - problem.solve() with defaults."""
    print("\n" + "=" * 60)
    print("Demo 1: Simplest Usage (Primary API)")
    print("=" * 60)

    problem = MFGProblem()

    # Primary API - one line solve
    result = problem.solve()

    print("\nSolved successfully")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final U error: {result.error_history_U[-1]:.2e}")
    print(f"  Final M error: {result.error_history_M[-1]:.2e}")
    if result.execution_time is not None:
        print(f"  Execution time: {result.execution_time:.3f}s")
    print(f"  Solution shapes: U={result.U.shape}, M={result.M.shape}")


def demo_custom_parameters():
    """Using custom parameters."""
    print("\n" + "=" * 60)
    print("Demo 2: Custom Parameters")
    print("=" * 60)

    # Geometry-First API: create grid, then problem
    geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[81])
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=30)

    # Custom solve parameters
    result = problem.solve(
        max_iterations=200,
        tolerance=1e-8,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error: {result.error_history_U[-1]:.2e}")


def demo_factory_api():
    """Advanced usage with factory API."""
    print("\n" + "=" * 60)
    print("Demo 3: Factory API (Advanced)")
    print("=" * 60)

    from mfg_pde.factory import create_standard_solver

    geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=20)

    # Factory API for more control
    solver = create_standard_solver(problem, "fixed_point")
    result = solver.solve(verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")


def demo_direct_solver():
    """Direct solver instantiation for full control."""
    print("\n" + "=" * 60)
    print("Demo 4: Direct Solver (Full Control)")
    print("=" * 60)

    from mfg_pde.alg.numerical.coupling import FixedPointIterator
    from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
    from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfg_pde.config.core import SolverConfig

    geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=20)

    # Create solvers directly
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPParticleSolver(problem)
    config = SolverConfig()

    # Create iterator
    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config,
    )

    result = iterator.solve(verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")


if __name__ == "__main__":
    print("MFG Problem Solving Demonstration")
    print("=" * 60)
    print("Primary API: problem.solve()")
    print("=" * 60)

    demo_simple_usage()
    demo_custom_parameters()
    demo_factory_api()
    demo_direct_solver()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
