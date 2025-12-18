#!/usr/bin/env python3
"""
Test to verify KDE mass conservation issue
"""

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver as ParticleFPSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import no_flux_bc


def test_kde_normalization():
    print("=== Testing KDE Mass Conservation ===")

    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.1, Nt=5, diffusion=1.0, coupling_coefficient=0.5)

    bc = no_flux_bc(dimension=1)

    print("Test 1: With normalize_kde_output=False (current)")
    solver1 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        normalize_kde_output=False,  # This is the issue!
        boundary_conditions=bc,
    )

    (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
    Nt_points = problem.Nt + 1  # Temporal grid points
    U_zero = np.zeros((Nt_points, Nx_points))
    M_result1 = solver1.solve_fp_system(M_initial=problem.m_init, drift_field=U_zero)

    dx = problem.geometry.get_grid_spacing()[0]
    mass1 = np.sum(M_result1 * dx, axis=1)
    print(f"  Initial mass: {mass1[0]:.6f}")
    print(f"  Final mass: {mass1[-1]:.6f}")
    print(f"  Mass loss: {mass1[0] - mass1[-1]:.6f} ({(mass1[0] - mass1[-1]) / mass1[0] * 100:.2f}%)")

    print("\nTest 2: With normalize_kde_output=True (should be better)")
    solver2 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        normalize_kde_output=True,  # This should help!
        boundary_conditions=bc,
    )

    M_result2 = solver2.solve_fp_system(M_initial=problem.m_init, drift_field=U_zero)

    mass2 = np.sum(M_result2 * dx, axis=1)
    print(f"  Initial mass: {mass2[0]:.6f}")
    print(f"  Final mass: {mass2[-1]:.6f}")
    print(f"  Mass loss: {mass2[0] - mass2[-1]:.6f} ({(mass2[0] - mass2[-1]) / mass2[0] * 100:.2f}%)")

    print("\nTest 3: Different bandwidth")
    solver3 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth=0.05,  # Fixed bandwidth
        normalize_kde_output=True,
        boundary_conditions=bc,
    )

    M_result3 = solver3.solve_fp_system(M_initial=problem.m_init, drift_field=U_zero)

    mass3 = np.sum(M_result3 * dx, axis=1)
    print(f"  Initial mass: {mass3[0]:.6f}")
    print(f"  Final mass: {mass3[-1]:.6f}")
    print(f"  Mass loss: {mass3[0] - mass3[-1]:.6f} ({(mass3[0] - mass3[-1]) / mass3[0] * 100:.2f}%)")


if __name__ == "__main__":
    test_kde_normalization()
