"""
Debug dimensional splitting to check shape mismatches.

Adds instrumentation to see what shapes are passed to 1D solver.
"""

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.alg.numerical.fp_solvers.fp_fdm_multid import solve_fp_nd_dimensional_splitting
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import BoundaryConditions


class Simple2DProblem(GridBasedMFGProblem):
    """Minimal 2D problem for debugging."""

    def __init__(self, N=8):
        super().__init__(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_resolution=N,
            time_domain=(0.1, 2),  # Small T, few steps
            diffusion_coeff=0.1,
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


# Monkey-patch _sweep_dimension to add logging
from mfg_pde.alg.numerical.fp_solvers import fp_fdm_multid

original_sweep = fp_fdm_multid._sweep_dimension


def instrumented_sweep(M_in, U_current, problem, dt, sweep_dim, boundary_conditions, backend):
    """Instrumented version that logs shapes."""
    print("\n_sweep_dimension called:")
    print(f"  sweep_dim: {sweep_dim}")
    print(f"  M_in.shape: {M_in.shape}")
    print(f"  U_current.shape: {U_current.shape}")
    print(f"  dt: {dt:.6f}")

    # Call original with some inspection
    ndim = problem.geometry.grid.dimension
    perp_dims = [d for d in range(ndim) if d != sweep_dim]

    # Look at first slice
    if len(perp_dims) == 0:
        M_slice = M_in
        U_slice = U_current
    else:
        # First slice (0, 0, ...)
        full_indices = [0] * ndim
        full_indices[sweep_dim] = slice(None)
        M_slice = M_in[tuple(full_indices)]
        U_slice = U_current[tuple(full_indices)]

    print(f"  M_slice.shape (first slice): {M_slice.shape}")
    print(f"  U_slice.shape (first slice): {U_slice.shape}")

    # Check what 1D solver will see
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm_multid import _FPProblem1DAdapter

    problem_1d = _FPProblem1DAdapter(
        full_problem=problem,
        sweep_dim=sweep_dim,
        fixed_indices=tuple([0] * len(perp_dims)),
        sweep_dt=dt,
    )

    print(f"  1D adapter Nx: {problem_1d.Nx}")
    print(f"  1D adapter expects input shape: ({problem_1d.Nx + 1},)")

    return original_sweep(M_in, U_current, problem, dt, sweep_dim, boundary_conditions, backend)


# Apply patch
fp_fdm_multid._sweep_dimension = instrumented_sweep


def main():
    print("=" * 70)
    print("  Debugging Dimensional Splitting Shape Mismatches")
    print("=" * 70)
    print()

    # Create 8x8 problem
    N = 8
    problem = Simple2DProblem(N=N)

    print(f"Problem: {N}×{N} grid")
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

    print("Running FP solver (will show shape information):")
    print()

    # Call solver
    boundary_conditions = BoundaryConditions(type="no_flux")
    M_solution = solve_fp_nd_dimensional_splitting(
        m_initial_condition=m0,
        U_solution_for_drift=U_zero,
        problem=problem,
        boundary_conditions=boundary_conditions,
        show_progress=False,
    )

    print()
    print("=" * 70)
    print(f"Solution shape: {M_solution.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
