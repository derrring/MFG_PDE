#!/usr/bin/env python3
"""
Debug the HJB solution from GFDM solver
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.hjb_solvers.gfdm_hjb import GFDMHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def debug_hjb_solution():
    print("=== Debugging GFDM HJB Solution ===")

    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=0.1, Nt=5, sigma=1.0, coefCT=0.5)

    # Create collocation points
    num_collocation_points = 10
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points - 1])
    no_flux_bc = BoundaryConditions(type='no_flux')

    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")

    # Create HJB solver
    hjb_solver = GFDMHJBSolver(
        problem=problem,
        collocation_points=collocation_points,
        delta=0.3,
        taylor_order=2,
        NiterNewton=10,
        l2errBoundNewton=1e-6,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
    )

    print(f"Collocation points: {num_collocation_points}")
    print(f"Boundary indices: {boundary_indices}")

    # Test with a simple density evolution (uniform)
    M_uniform = np.ones((problem.Nt + 1, problem.Nx + 1))
    for t in range(problem.Nt + 1):
        M_uniform[t, :] = problem.m_init / np.sum(problem.m_init * problem.Dx)

    print(f"Test density mass: {np.sum(M_uniform[0,:] * problem.Dx):.6f}")

    # Set simple terminal condition
    U_terminal = np.zeros(problem.Nx + 1)

    # Initial guess for U
    U_initial = np.zeros((problem.Nt + 1, problem.Nx + 1))

    try:
        print("\nSolving HJB system...")
        U_solution = hjb_solver.solve_hjb_system(
            M_density_evolution_from_FP=M_uniform, U_final_condition_at_T=U_terminal, U_from_prev_picard=U_initial
        )

        if U_solution is not None:
            print(f"U_solution shape: {U_solution.shape}")
            print(f"U min/max at t=0: [{np.min(U_solution[0,:]):.6f}, {np.max(U_solution[0,:]):.6f}]")
            print(f"U min/max at t=T: [{np.min(U_solution[-1,:]):.6f}, {np.max(U_solution[-1,:]):.6f}]")

            # Check for extreme values
            if np.any(np.abs(U_solution) > 1e6):
                print("WARNING: U solution has extreme values!")
                print(f"  Absolute max: {np.max(np.abs(U_solution)):.2e}")

            # Check for NaN or Inf
            if np.any(np.isnan(U_solution)) or np.any(np.isinf(U_solution)):
                print("ERROR: U solution contains NaN or Inf!")
            else:
                print("OK: U solution is finite")

            # Plot the solution at t=0
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(problem.xSpace, U_solution[0, :], 'b-', label='U at t=0')
            plt.xlabel('x')
            plt.ylabel('U(0,x)')
            plt.title('HJB Solution at t=0')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(problem.xSpace, M_uniform[0, :], 'r-', label='M at t=0')
            plt.xlabel('x')
            plt.ylabel('M(0,x)')
            plt.title('Density at t=0')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

        else:
            print("ERROR: HJB solver returned None!")

    except Exception as e:
        print(f"HJB solver failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_hjb_solution()
