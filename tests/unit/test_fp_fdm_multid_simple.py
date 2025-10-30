"""
Simplified tests for multi-dimensional FP FDM solver.
"""

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import BoundaryConditions


class SimpleFPTestProblem(GridBasedMFGProblem):
    """Simple test problem for FP solver validation."""

    def __init__(self, domain_bounds, grid_resolution, time_domain=(1.0, 100), diffusion_coeff=0.1):
        super().__init__(domain_bounds, grid_resolution, time_domain, diffusion_coeff)

    def initial_density(self, x):
        return np.ones(x.shape[0])

    def terminal_cost(self, x):
        return np.zeros(x.shape[0])

    def running_cost(self, x, t):
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared

    def setup_components(self):
        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            return 0.0

        def initial_density_func(x_idx):
            return 1.0

        def terminal_cost_func(x_idx):
            return 0.0

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def test_fp_1d():
    """Test 1D FP solver."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0),
        grid_resolution=20,
        time_domain=(0.1, 10),
        diffusion_coeff=0.1,
    )
    solver = FPFDMSolver(problem)
    assert solver.dimension == 1

    # For 1D, original solver expects (Nx+1,) arrays
    # Grid has num_points grid points, arrays include both boundaries
    Nt = problem.Nt + 1
    Nx = problem.geometry.grid.num_points[0]
    U_zero = np.zeros((Nt, Nx))
    m_init = np.ones(Nx) / Nx

    M_solution = solver.solve_fp_system(m_init, U_zero, show_progress=False)

    assert M_solution.shape == (Nt, Nx)
    assert np.all(M_solution >= -1e-10)
    assert not np.any(np.isnan(M_solution))
    print("✓ 1D FP solver works")


def test_fp_2d():
    """Test 2D FP solver."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=10,
        time_domain=(0.05, 10),
        diffusion_coeff=0.1,
    )
    solver = FPFDMSolver(problem)
    assert solver.dimension == 2

    # For 2D+, arrays have shape (N1-1, N2-1, ...) - exclude right boundaries
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))
    U_zero = np.zeros((Nt, *shape))
    m_init = np.ones(shape) / np.prod(shape)

    M_solution = solver.solve_fp_system(m_init, U_zero, show_progress=False)

    expected_shape = (Nt, *shape)
    assert M_solution.shape == expected_shape
    assert np.all(M_solution >= -1e-10)
    assert not np.any(np.isnan(M_solution))
    print(f"✓ 2D FP solver works (shape={expected_shape})")


def test_fp_3d():
    """Test 3D FP solver."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        grid_resolution=6,
        time_domain=(0.02, 5),
        diffusion_coeff=0.1,
    )
    solver = FPFDMSolver(problem)
    assert solver.dimension == 3

    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(3))
    U_zero = np.zeros((Nt, *shape))
    m_init = np.ones(shape) / np.prod(shape)

    M_solution = solver.solve_fp_system(m_init, U_zero, show_progress=False)

    expected_shape = (Nt, *shape)
    assert M_solution.shape == expected_shape
    assert np.all(M_solution >= -1e-10)
    assert not np.any(np.isnan(M_solution))
    print(f"✓ 3D FP solver works (shape={expected_shape})")


def test_fp_2d_mass_conservation():
    """Test mass conservation in 2D."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=10,
        time_domain=(0.05, 10),
        diffusion_coeff=0.1,
    )
    solver = FPFDMSolver(problem, boundary_conditions=BoundaryConditions(type="no_flux"))

    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))
    U_zero = np.zeros((Nt, *shape))
    m_init = np.ones(shape)
    m_init = m_init / np.sum(m_init)

    M_solution = solver.solve_fp_system(m_init, U_zero, show_progress=False)

    dx, dy = problem.geometry.grid.spacing
    cell_volume = dx * dy
    initial_mass = np.sum(m_init) * cell_volume
    final_mass = np.sum(M_solution[-1]) * cell_volume
    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)

    print(f"✓ 2D mass conservation: initial={initial_mass:.6f}, final={final_mass:.6f}, error={mass_error:.2%}")
    assert mass_error < 0.10, f"Mass error {mass_error:.2%} too large"


if __name__ == "__main__":
    test_fp_1d()
    test_fp_2d()
    test_fp_3d()
    test_fp_2d_mass_conservation()
    print("\n✅ All FP tests passed!")
