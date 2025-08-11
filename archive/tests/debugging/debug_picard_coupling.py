#!/usr/bin/env python3
"""
Debug the Picard coupling in particle-collocation method
"""

import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def debug_picard_coupling():
    print("=== Debugging Picard Coupling in Particle-Collocation ===")

    # Very simple problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.05, Nt=3, sigma=1.0, coefCT=0.5)

    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")

    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=50,
        delta=0.5,
        taylor_order=2,
        kde_bandwidth="scott",
        normalize_kde_output=True,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")

    # Test just 2 Picard iterations to see what happens
    try:
        U_solution, M_solution, solve_info = solver.solve(Niter=3, l2errBound=1e-3, verbose=True)  # Very few iterations

        if M_solution is not None:
            mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
            print(f"\nMass evolution over time:")
            for i, mass in enumerate(mass_evolution):
                print(f"  t[{i}]: {mass:.6f}")

            print(f"\nFinal mass change: {mass_evolution[-1] - mass_evolution[0]:.6f}")
            print(f"Relative mass change: {(mass_evolution[-1] - mass_evolution[0])/mass_evolution[0]*100:.2f}%")

        else:
            print("M_solution is None!")

    except Exception as e:
        print(f"Solver failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_picard_coupling()
