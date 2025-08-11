#!/usr/bin/env python3
"""
Test GFDM solver with very simple parameters
"""

import numpy as np

from mfg_pde.alg.hjb_solvers.gfdm_hjb import GFDMHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_simple_gfdm():
    print("=== Testing GFDM with Very Simple Parameters ===")

    # Very simple problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.02, Nt=2, sigma=0.1, coefCT=0.1)

    # Few collocation points
    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation points: {collocation_points.flatten()}")

    # Test with Dirichlet boundary conditions first
    print("\n=== Test 1: Dirichlet Boundary Conditions ===")
    dirichlet_bc = {"type": "dirichlet", "value": 0.0}

    hjb_solver_dirichlet = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.8,  # Large delta to include all points
        taylor_order=1,  # Simple first order
        NiterNewton=5,
        l2errBoundNewton=1e-3,
        boundary_indices=boundary_indices,
        boundary_conditions=dirichlet_bc,
    )

    # Simple uniform density
    M_simple = np.ones((problem.Nt + 1, problem.Nx + 1)) * 0.5
    U_terminal = np.zeros(problem.Nx + 1)
    U_initial = np.zeros((problem.Nt + 1, problem.Nx + 1))

    try:
        U_dirichlet = hjb_solver_dirichlet.solve_hjb_system(
            M_density_evolution_from_FP=M_simple, U_final_condition_at_T=U_terminal, U_from_prev_picard=U_initial
        )

        print(f"Dirichlet success: U range = [{np.min(U_dirichlet):.3f}, {np.max(U_dirichlet):.3f}]")
        print(f"  Max absolute value: {np.max(np.abs(U_dirichlet)):.3f}")

        if np.any(np.isnan(U_dirichlet)) or np.any(np.isinf(U_dirichlet)):
            print("  ERROR: Solution contains NaN or Inf")
        elif np.max(np.abs(U_dirichlet)) > 1e6:
            print("  WARNING: Solution has extreme values")
        else:
            print("  OK: Solution seems reasonable")

    except Exception as e:
        print(f"Dirichlet failed: {e}")
        import traceback

        traceback.print_exc()

    # Test with no-flux boundary conditions
    print("\n=== Test 2: No-Flux Boundary Conditions ===")
    no_flux_bc = BoundaryConditions(type='no_flux')

    hjb_solver_noflux = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.8,
        taylor_order=1,
        NiterNewton=5,
        l2errBoundNewton=1e-3,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    try:
        U_noflux = hjb_solver_noflux.solve_hjb_system(
            M_density_evolution_from_FP=M_simple, U_final_condition_at_T=U_terminal, U_from_prev_picard=U_initial
        )

        print(f"No-flux success: U range = [{np.min(U_noflux):.3f}, {np.max(U_noflux):.3f}]")
        print(f"  Max absolute value: {np.max(np.abs(U_noflux)):.3f}")

        if np.any(np.isnan(U_noflux)) or np.any(np.isinf(U_noflux)):
            print("  ERROR: Solution contains NaN or Inf")
        elif np.max(np.abs(U_noflux)) > 1e6:
            print("  WARNING: Solution has extreme values")
        else:
            print("  OK: Solution seems reasonable")

        # Compare with Dirichlet
        if 'U_dirichlet' in locals():
            diff = np.max(np.abs(U_noflux - U_dirichlet))
            print(f"  Difference from Dirichlet: {diff:.3f}")

    except Exception as e:
        print(f"No-flux failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_simple_gfdm()
