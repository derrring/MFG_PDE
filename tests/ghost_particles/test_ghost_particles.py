#!/usr/bin/env python3
"""
Test ghost particle implementation for no-flux boundary conditions
"""

import numpy as np

from mfg_pde.alg.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_ghost_particles():
    print("=== Testing Ghost Particle Implementation ===")

    # Simple problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.02, Nt=2, sigma=0.1, coefCT=0.1)

    # Create collocation points including boundary points
    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])  # First and last points
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Collocation points: {collocation_points.flatten()}")
    print(f"Boundary indices: {boundary_indices}")

    # Create GFDM solver with ghost particles
    hjb_solver = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.8,  # Large delta to include many neighbors
        taylor_order=1,
        weight_function="wendland",
        NiterNewton=5,
        l2errBoundNewton=1e-3,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    # Examine neighborhood structure for boundary points
    print(f"\n=== Neighborhood Analysis ===")
    for i in [0, num_collocation_points - 1]:  # Boundary points
        neighborhood = hjb_solver.neighborhoods[i]
        print(f"\nBoundary point {i} at x = {collocation_points[i, 0]:.3f}:")
        print(f"  Total neighbors: {neighborhood['size']}")
        print(f"  Has ghost particles: {neighborhood['has_ghost']}")
        print(f"  Ghost count: {neighborhood['ghost_count']}")
        print(f"  Neighbor indices: {neighborhood['indices']}")
        print(f"  Neighbor points: {neighborhood['points']}")
        print(f"  Neighbor distances: {neighborhood['distances']}")

    # Test derivative approximation with a known function
    print(f"\n=== Testing Derivative Approximation ===")

    # Test with a quadratic function u(x) = x^2
    # Analytical: u'(0) = 0, u'(1) = 2, u''(x) = 2 everywhere
    u_test_quadratic = collocation_points.flatten() ** 2

    print(f"Test function: u(x) = x^2")
    print(f"Function values: {u_test_quadratic}")

    # Test derivatives at boundary points
    for boundary_idx in [0, num_collocation_points - 1]:
        x = collocation_points[boundary_idx, 0]
        derivatives = hjb_solver.approximate_derivatives(u_test_quadratic, boundary_idx)

        analytical_first = 2 * x  # u'(x) = 2x
        analytical_second = 2.0  # u''(x) = 2

        print(f"\nBoundary point {boundary_idx} at x = {x:.3f}:")

        if (1,) in derivatives:
            numerical_first = derivatives[(1,)]
            error_first = abs(numerical_first - analytical_first)
            print(
                f"  First derivative: numerical={numerical_first:.6f}, analytical={analytical_first:.6f}, error={error_first:.2e}"
            )

        if (2,) in derivatives:
            numerical_second = derivatives[(2,)]
            error_second = abs(numerical_second - analytical_second)
            print(
                f"  Second derivative: numerical={numerical_second:.6f}, analytical={analytical_second:.6f}, error={error_second:.2e}"
            )

    # Test with a function that should have zero derivative at boundaries
    print(f"\n=== Testing with Zero-Derivative Function ===")

    # Test with u(x) = 1 - 2*x + x^2 = (x-1)^2
    # This has u'(0) = -2, u'(1) = 0, which violates no-flux at left boundary
    # Let's test with a symmetric function: u(x) = (x - 0.5)^2
    # This has u'(0) = -1, u'(1) = 1, still not zero at boundaries

    # Better test: u(x) = cos(π*x) which has u'(0) = 0, u'(1) = 0 naturally
    u_test_cosine = np.cos(np.pi * collocation_points.flatten())

    print(f"Test function: u(x) = cos(π*x)")
    print(f"Function values: {u_test_cosine}")

    for boundary_idx in [0, num_collocation_points - 1]:
        x = collocation_points[boundary_idx, 0]
        derivatives = hjb_solver.approximate_derivatives(u_test_cosine, boundary_idx)

        analytical_first = -np.pi * np.sin(np.pi * x)  # Should be 0 at both boundaries
        analytical_second = -np.pi**2 * np.cos(np.pi * x)

        print(f"\nBoundary point {boundary_idx} at x = {x:.3f}:")

        if (1,) in derivatives:
            numerical_first = derivatives[(1,)]
            error_first = abs(numerical_first - analytical_first)
            print(
                f"  First derivative: numerical={numerical_first:.6f}, analytical={analytical_first:.6f}, error={error_first:.2e}"
            )

            if abs(numerical_first) < 1e-6:
                print(f"  ✓ No-flux condition satisfied!")
            else:
                print(f"  ⚠ No-flux condition violated")

    # Test HJB solution
    print(f"\n=== Testing HJB Solution ===")
    M_simple = np.ones((problem.Nt + 1, problem.Nx + 1)) * 0.5
    U_terminal = np.zeros(problem.Nx + 1)
    U_initial = np.zeros((problem.Nt + 1, problem.Nx + 1))

    try:
        U_solution = hjb_solver.solve_hjb_system(
            M_density_evolution_from_FP=M_simple, U_final_condition_at_T=U_terminal, U_from_prev_picard=U_initial
        )

        max_val = np.max(np.abs(U_solution))
        print(f"HJB solution success: Max |U| = {max_val:.3f}")

        # Check if boundary conditions are satisfied
        u_at_boundaries = U_solution[0, [0, -1]]  # U at t=0, boundaries
        print(f"U at boundaries (t=0): left={u_at_boundaries[0]:.6f}, right={u_at_boundaries[1]:.6f}")

        if max_val < 1e6:
            print(f"✓ Solution seems reasonable")
        else:
            print(f"⚠ Solution has large values")

    except Exception as e:
        print(f"❌ HJB solution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_ghost_particles()
