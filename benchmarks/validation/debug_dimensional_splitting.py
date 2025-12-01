"""
Debug 2D FP FDM solver for shape and mass conservation checks.

Note: The original dimensional splitting solver (fp_fdm_multid) has been removed.
This file now tests the unified FPFDMSolver instead.
"""

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.geometry.boundary import no_flux_bc


class Simple2DProblem(MFGProblem):
    """Minimal 2D problem for debugging."""

    def __init__(self, N=8):
        super().__init__(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[N, N],
            T=0.1,
            Nt=2,
            sigma=0.1,
        )

    def initial_density(self, x):
        """Uniform density."""
        return np.ones(x.shape[0]) / x.shape[0]

    def terminal_cost(self, x):
        return np.zeros(x.shape[0])

    def running_cost(self, x, t):
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        return np.zeros(x.shape[0])

    def setup_components(self):
        return MFGComponents(
            hamiltonian_func=lambda *args, **kwargs: 0.0,
            hamiltonian_dm_func=lambda *args, **kwargs: 0.0,
            initial_density_func=lambda *args, **kwargs: 0.0,
            final_value_func=lambda *args, **kwargs: 0.0,
        )


def main():
    print("=" * 70)
    print("  Debug 2D FP FDM Solver")
    print("=" * 70)
    print()

    # Create 8x8 problem
    N = 8
    problem = Simple2DProblem(N=N)

    print(f"Problem: {N}x{N} grid")
    print(f"Grid shape: {tuple(problem.geometry.grid.num_points)}")
    print(f"Grid spacing: {problem.geometry.grid.spacing}")
    print()

    # Get initial density
    grid_points = problem.geometry.grid.flatten()
    shape = tuple(problem.geometry.grid.num_points)
    m0 = problem.initial_density(grid_points).reshape(shape)

    print(f"Initial density shape: {m0.shape}")
    print()

    # Zero velocity field
    U_zero = np.zeros((problem.Nt + 1, *shape))

    print(f"Value function shape: {U_zero.shape}")
    print()

    print("Running FP FDM solver:")
    print()

    # Call solver
    boundary_conditions = no_flux_bc(dimension=2)
    fp_solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)
    M_solution = fp_solver.solve_fp_system(
        M_initial=m0,
        drift_field=U_zero,
        show_progress=False,
    )

    print()
    print("=" * 70)
    print(f"Solution shape: {M_solution.shape}")

    # Check mass conservation
    dV = float(np.prod(problem.geometry.grid.spacing))
    mass_initial = np.sum(M_solution[0]) * dV
    mass_final = np.sum(M_solution[-1]) * dV
    print(f"Initial mass: {mass_initial:.10f}")
    print(f"Final mass: {mass_final:.10f}")
    print(f"Mass change: {(mass_final - mass_initial) / mass_initial * 100:.6f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
