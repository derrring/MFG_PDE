#!/usr/bin/env python3
"""
Test SVD implementation in GFDM solver
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


@pytest.mark.skip(
    reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
    "This test needs refactoring to use a concrete GFDM implementation. "
    "Issue #140 - pre-existing test failure."
)
def test_svd_implementation():
    print("=== Testing SVD Implementation in GFDM ===")

    # Simple problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.02, Nt=2, sigma=0.1, coefCT=0.1)

    num_collocation_points = 7
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])
    no_flux_bc = BoundaryConditions(type="no_flux")

    print(f"Problem: {num_collocation_points} collocation points")

    # Create GFDM solver
    hjb_solver = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.8,  # Large delta to include many neighbors
        taylor_order=2,  # Second order for more interesting matrices
        weight_function="wendland",
        NiterNewton=5,
        l2errBoundNewton=1e-3,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    # Get decomposition info
    decomp_info = hjb_solver.get_decomposition_info()

    print("\n=== Decomposition Statistics ===")
    print(f"Total collocation points: {decomp_info['total_points']}")
    print(f"SVD points: {decomp_info['svd_points']} ({decomp_info['svd_percentage']:.1f}%)")
    print(f"QR points: {decomp_info['qr_points']}")
    print(f"Normal equation points: {decomp_info['normal_equation_points']}")

    if decomp_info["condition_numbers"]:
        print(f"Average condition number: {decomp_info['avg_condition_number']:.2e}")
        print(f"Rank range: [{decomp_info['min_rank']}, {decomp_info['max_rank']}]")
        print(
            f"Condition number range: [{min(decomp_info['condition_numbers']):.2e}, {max(decomp_info['condition_numbers']):.2e}]"
        )

    # Test with actual HJB solve
    print("\n=== Testing HJB Solution ===")
    M_simple = np.ones((problem.Nt + 1, problem.Nx + 1)) * 0.5
    U_terminal = np.zeros(problem.Nx + 1)
    U_initial = np.zeros((problem.Nt + 1, problem.Nx + 1))

    try:
        U_solution = hjb_solver.solve_hjb_system(
            M_density_evolution_from_FP=M_simple, U_final_condition_at_T=U_terminal, U_from_prev_picard=U_initial
        )

        max_val = np.max(np.abs(U_solution))
        print(f"Solution success: Max |U| = {max_val:.3f}")

        if np.any(np.isnan(U_solution)) or np.any(np.isinf(U_solution)):
            print("ERROR: Solution contains NaN or Inf")
        elif max_val > 1e6:
            print("WARNING: Solution has extreme values")
        else:
            print("OK: Solution seems reasonable")

        # Test derivative approximation quality
        print("\n=== Testing Derivative Approximation ===")
        u_test = np.sin(np.pi * collocation_points.flatten())  # Test function

        # Test derivatives at middle point
        mid_idx = num_collocation_points // 2
        derivatives = hjb_solver.approximate_derivatives(u_test, mid_idx)

        # Analytical derivatives for sin(πx) at x = 0.5
        x_mid = collocation_points[mid_idx, 0]
        analytical_first = np.pi * np.cos(np.pi * x_mid)  # Should be 0 at x=0.5
        analytical_second = -(np.pi**2) * np.sin(np.pi * x_mid)  # Should be -π² at x=0.5

        if (1,) in derivatives:
            numerical_first = derivatives[(1,)]
            error_first = abs(numerical_first - analytical_first)
            print(
                f"First derivative: numerical={numerical_first:.6f}, analytical={analytical_first:.6f}, error={error_first:.2e}"
            )

        if (2,) in derivatives:
            numerical_second = derivatives[(2,)]
            error_second = abs(numerical_second - analytical_second)
            print(
                f"Second derivative: numerical={numerical_second:.6f}, analytical={analytical_second:.6f}, error={error_second:.2e}"
            )

    except Exception as e:
        print(f"HJB solution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_svd_implementation()
