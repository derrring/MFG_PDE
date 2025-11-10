#!/usr/bin/env python3
"""
Test to verify KDE mass conservation issue
"""

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver as ParticleFPSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions


def test_kde_normalization():
    print("=== Testing KDE Mass Conservation ===")

    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.1, Nt=5, sigma=1.0, coupling_coefficient=0.5)

    no_flux_bc = BoundaryConditions(type="no_flux")

    print("Test 1: With normalize_kde_output=False (current)")
    solver1 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        normalize_kde_output=False,  # This is the issue!
        boundary_conditions=no_flux_bc,
    )

    U_zero = np.zeros((problem.Nt + 1, problem.Nx + 1))
    M_result1 = solver1.solve_fp_system(m_initial_condition=problem.m_init, U_solution_for_drift=U_zero)

    mass1 = np.sum(M_result1 * problem.dx, axis=1)
    print(f"  Initial mass: {mass1[0]:.6f}")
    print(f"  Final mass: {mass1[-1]:.6f}")
    print(f"  Mass loss: {mass1[0] - mass1[-1]:.6f} ({(mass1[0] - mass1[-1]) / mass1[0] * 100:.2f}%)")

    print("\nTest 2: With normalize_kde_output=True (should be better)")
    solver2 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        normalize_kde_output=True,  # This should help!
        boundary_conditions=no_flux_bc,
    )

    M_result2 = solver2.solve_fp_system(m_initial_condition=problem.m_init, U_solution_for_drift=U_zero)

    mass2 = np.sum(M_result2 * problem.dx, axis=1)
    print(f"  Initial mass: {mass2[0]:.6f}")
    print(f"  Final mass: {mass2[-1]:.6f}")
    print(f"  Mass loss: {mass2[0] - mass2[-1]:.6f} ({(mass2[0] - mass2[-1]) / mass2[0] * 100:.2f}%)")

    print("\nTest 3: Different bandwidth")
    solver3 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth=0.05,  # Fixed bandwidth
        normalize_kde_output=True,
        boundary_conditions=no_flux_bc,
    )

    M_result3 = solver3.solve_fp_system(m_initial_condition=problem.m_init, U_solution_for_drift=U_zero)

    mass3 = np.sum(M_result3 * problem.dx, axis=1)
    print(f"  Initial mass: {mass3[0]:.6f}")
    print(f"  Final mass: {mass3[-1]:.6f}")
    print(f"  Mass loss: {mass3[0] - mass3[-1]:.6f} ({(mass3[0] - mass3[-1]) / mass3[0] * 100:.2f}%)")


if __name__ == "__main__":
    test_kde_normalization()
