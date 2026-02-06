#!/usr/bin/env python3
"""
Test different weight functions in GFDM solver
"""

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def test_weight_functions():
    print("=== Testing Different Weight Functions in GFDM ===")

    # Simple problem
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=10 intervals
    problem = MFGProblem(
        geometry=geometry, T=0.02, Nt=2, diffusion=0.1, coupling_coefficient=0.1, components=_default_components()
    )

    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])
    bc = no_flux_bc(dimension=1)

    # Test data
    (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
    Nt_points = problem.Nt + 1  # Temporal grid points
    M_simple = np.ones((Nt_points, Nx_points)) * 0.5
    U_terminal = np.zeros(Nx_points)
    U_initial = np.zeros((Nt_points, Nx_points))

    weight_functions = ["uniform", "gaussian", "wendland", "inverse_distance"]
    results = {}

    for weight_func in weight_functions:
        print(f"\n=== Testing {weight_func} weight function ===")

        try:
            hjb_solver = GFDMHJBSolver(
                problem=problem,
                collocation_points=collocation_points,
                delta=0.8,
                taylor_order=1,
                weight_function=weight_func,
                max_newton_iterations=5,
                newton_tolerance=1e-3,
                boundary_indices=boundary_indices,
                boundary_conditions=bc,
            )

            U_solution = hjb_solver.solve_hjb_system(
                M_density=M_simple, U_terminal=U_terminal, U_coupling_prev=U_initial
            )

            max_val = np.max(np.abs(U_solution))
            results[weight_func] = max_val

            print(f"  Success: Max |U| = {max_val:.3f}")

            if np.any(np.isnan(U_solution)) or np.any(np.isinf(U_solution)):
                print("  ERROR: Contains NaN or Inf")
            elif max_val > 1e6:
                print("  WARNING: Extreme values")
            else:
                print("  OK: Reasonable values")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[weight_func] = None

    print("\n=== Weight Function Comparison ===")
    print(f"{'Function':<15} {'Max |U|':<10} {'Status'}")
    print(f"{'-' * 15} {'-' * 10} {'-' * 10}")

    for func, max_val in results.items():
        if max_val is not None:
            status = "OK" if max_val < 100 else "High" if max_val < 1e6 else "Extreme"
            print(f"{func:<15} {max_val:<10.3f} {status}")
        else:
            print(f"{func:<15} {'FAILED':<10} {'ERROR'}")

    # Find the best weight function (lowest max value)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_func = min(valid_results.keys(), key=lambda k: valid_results[k])
        print(f"\nBest weight function: {best_func} (Max |U| = {valid_results[best_func]:.3f})")


if __name__ == "__main__":
    test_weight_functions()
