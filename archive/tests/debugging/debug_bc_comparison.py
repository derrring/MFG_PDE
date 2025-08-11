#!/usr/bin/env python3
"""
Compare particle-collocation with different boundary conditions
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_boundary_conditions():
    print("=== Testing Particle-Collocation with Different Boundary Conditions ===")

    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.2, Nt=10, sigma=1.0, coefCT=0.5)

    num_collocation_points = 10
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")

    # Test 1: Dirichlet boundary conditions (should work)
    print(f"\n=== Test 1: Dirichlet Boundary Conditions ===")
    dirichlet_bc = {"type": "dirichlet", "value": 0.0}  # Dictionary format

    try:
        solver1 = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.3,
            taylor_order=2,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=boundary_indices,
            boundary_conditions=dirichlet_bc,
        )

        U1, M1, info1 = solver1.solve(Niter=5, l2errBound=1e-3, verbose=False)

        if M1 is not None:
            mass1 = np.sum(M1 * problem.Dx, axis=1)
            print(f"  Success: Mass change = {mass1[-1] - mass1[0]:.6f}")
            print(f"  Final mass: {mass1[-1]:.6f}")
        else:
            print(f"  Failed: M1 is None")

    except Exception as e:
        print(f"  Error with Dirichlet BC: {e}")

    # Test 2: No-flux boundary conditions (problematic)
    print(f"\n=== Test 2: No-Flux Boundary Conditions ===")
    no_flux_bc = BoundaryConditions(type='no_flux')

    try:
        solver2 = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.3,
            taylor_order=2,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
        )

        U2, M2, info2 = solver2.solve(Niter=5, l2errBound=1e-3, verbose=False)

        if M2 is not None:
            mass2 = np.sum(M2 * problem.Dx, axis=1)
            print(f"  Mass evolution: {[f'{m:.3f}' for m in mass2]}")
            print(f"  Final mass: {mass2[-1]:.6f}")
            print(f"  Mass change: {mass2[-1] - mass2[0]:.6f}")

            # Check if solution becomes zero
            if mass2[-1] < 1e-6:
                print(f"  ERROR: Solution collapsed to zero!")
            else:
                print(f"  OK: Solution preserved mass")
        else:
            print(f"  Failed: M2 is None")

    except Exception as e:
        print(f"  Error with No-flux BC: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_boundary_conditions()
