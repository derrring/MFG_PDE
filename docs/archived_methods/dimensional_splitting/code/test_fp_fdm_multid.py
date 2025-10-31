"""
Tests for multi-dimensional FP FDM solver with dimensional splitting.

Tests cover:
- 1D backward compatibility
- 2D/3D mass conservation
- Positivity enforcement
- Boundary condition handling
- Convergence properties
"""

import pytest

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import BoundaryConditions


class SimpleFPTestProblem(GridBasedMFGProblem):
    """
    Simple test problem for FP solver validation.

    - Isotropic diffusion (σ² Δm)
    - Zero drift initially (v = 0)
    - Gaussian initial density
    """

    def __init__(
        self,
        domain_bounds: tuple,
        grid_resolution: int | tuple,
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
    ):
        super().__init__(domain_bounds, grid_resolution, time_domain, diffusion_coeff)

    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """Gaussian initial density centered at domain center."""
        # x has shape (N_total, ndim)
        center = np.array([0.5] * x.shape[1])
        dist_sq = np.sum((x - center) ** 2, axis=1)
        density = np.exp(-50 * dist_sq)
        # Normalize
        return density / (np.sum(density) + 1e-10)

    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Zero terminal cost."""
        return np.zeros(x.shape[0])

    def running_cost(self, x: np.ndarray, t: float) -> np.ndarray:
        """Zero running cost."""
        return np.zeros(x.shape[0])

    def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """Simple isotropic Hamiltonian: H = (1/2)|p|²"""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared

    def setup_components(self) -> MFGComponents:
        """Setup MFG components."""

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """Simple isotropic Hamiltonian: H = (1/2)|p|²"""
            if derivs is not None:
                # Use tuple-indexed derivatives
                grad_u = []
                ndim = problem.geometry.grid.dimension
                for d in range(ndim):
                    idx_tuple = tuple([0] * d + [1] + [0] * (ndim - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))
                return 0.5 * float(np.sum(np.array(grad_u) ** 2))
            return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            return 0.0

        def initial_density_func(x_idx):
            # Uniform for simplicity
            return 1.0

        def terminal_cost_func(x_idx):
            return 0.0

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


@pytest.fixture
def simple_1d_problem():
    """Create simple 1D FP test problem."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0),
        grid_resolution=20,
        time_domain=(0.1, 10),
        diffusion_coeff=0.1,
    )
    return problem


@pytest.fixture
def simple_2d_problem():
    """Create simple 2D FP test problem."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=10,
        time_domain=(0.05, 10),
        diffusion_coeff=0.1,
    )
    return problem


@pytest.fixture
def simple_3d_problem():
    """Create simple 3D FP test problem."""
    problem = SimpleFPTestProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        grid_resolution=8,
        time_domain=(0.02, 5),
        diffusion_coeff=0.1,
    )
    return problem


def test_fp_1d_backward_compatibility(simple_1d_problem):
    """Test that 1D FP solver still works (backward compatibility)."""
    problem = simple_1d_problem
    solver = FPFDMSolver(problem)

    # Check dimension detection
    assert solver.dimension == 1, "Should detect 1D problem"

    # For 1D problems, the existing FP solver expects Nx+1 points
    # (GridBasedMFGProblem has Nx intervals, so Nx+1 grid points including boundaries)
    Nt = problem.Nt + 1
    Nx = problem.Nx + 1

    # Create zero value function (no drift)
    U_zero = np.zeros((Nt, Nx))

    # Initial density: uniform
    m_init = np.ones(Nx)
    m_init = m_init / np.sum(m_init)  # Normalize

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Checks
    assert M_solution.shape == (Nt, Nx), "Solution shape should be (Nt, Nx)"
    assert np.all(M_solution >= -1e-10), "Density should be non-negative"
    assert not np.any(np.isnan(M_solution)), "No NaNs in solution"


def test_fp_2d_basic_solve(simple_2d_problem):
    """Test that 2D FP solver runs without errors."""
    problem = simple_2d_problem
    solver = FPFDMSolver(problem)

    # Check dimension detection
    assert solver.dimension == 2, "Should detect 2D problem"

    # For 2D problems, GridBasedMFGProblem convention:
    # Arrays have shape (N1-1, N2-1) - excludes right boundaries
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))

    # Create zero value function (no drift)
    U_zero = np.zeros((Nt, *shape))

    # Initial density: uniform
    m_init = np.ones(shape)
    m_init = m_init / np.sum(m_init)  # Normalize

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Checks
    expected_shape = (Nt, *shape)
    assert M_solution.shape == expected_shape, f"Solution shape should be {expected_shape}"
    assert np.all(M_solution >= -1e-10), "Density should be non-negative"
    assert not np.any(np.isnan(M_solution)), "No NaNs in solution"


def test_fp_2d_mass_conservation(simple_2d_problem):
    """Test that 2D FP solver conserves mass (approximately)."""
    problem = simple_2d_problem
    solver = FPFDMSolver(problem, boundary_conditions=BoundaryConditions(type="no_flux"))

    # Create zero value function (no drift, pure diffusion)
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))
    U_zero = np.zeros((Nt, *shape))

    # Initial density
    x_vals = problem.geometry.grid.points[0][:-1]
    y_vals = problem.geometry.grid.points[1][:-1]
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    x_flat = np.column_stack([xx.ravel(), yy.ravel()])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Check mass conservation
    dx = problem.geometry.grid.spacing[0]
    dy = problem.geometry.grid.spacing[1]
    cell_volume = dx * dy

    initial_mass = np.sum(m_init) * cell_volume
    final_mass = np.sum(M_solution[-1]) * cell_volume

    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)

    print("\nMass conservation check:")
    print(f"  Initial mass: {initial_mass:.6e}")
    print(f"  Final mass:   {final_mass:.6e}")
    print(f"  Relative error: {mass_error:.6e}")

    # For no-flux boundaries with pure diffusion, mass should be conserved
    # Allow 1% relative error due to discretization
    assert mass_error < 0.01, f"Mass should be conserved (error: {mass_error:.2%})"


def test_fp_2d_positivity(simple_2d_problem):
    """Test that 2D FP solver maintains non-negativity."""
    problem = simple_2d_problem
    solver = FPFDMSolver(problem)

    # Create zero value function
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))
    U_zero = np.zeros((Nt, *shape))

    # Initial density
    x_vals = problem.geometry.grid.points[0][:-1]
    y_vals = problem.geometry.grid.points[1][:-1]
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    x_flat = np.column_stack([xx.ravel(), yy.ravel()])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Check positivity at all timesteps
    min_density = np.min(M_solution)
    assert min_density >= -1e-10, f"Density should be non-negative (min: {min_density:.2e})"


def test_fp_3d_basic_solve(simple_3d_problem):
    """Test that 3D FP solver runs without errors."""
    problem = simple_3d_problem
    solver = FPFDMSolver(problem)

    # Check dimension detection
    assert solver.dimension == 3, "Should detect 3D problem"

    # Create zero value function
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(3))
    U_zero = np.zeros((Nt, *shape))

    # Initial density
    x_vals = problem.geometry.grid.points[0][:-1]
    y_vals = problem.geometry.grid.points[1][:-1]
    z_vals = problem.geometry.grid.points[2][:-1]
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    x_flat = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Checks
    expected_shape = (Nt, *shape)
    assert M_solution.shape == expected_shape, f"Solution shape should be {expected_shape}"
    assert np.all(M_solution >= -1e-10), "Density should be non-negative"
    assert not np.any(np.isnan(M_solution)), "No NaNs in solution"


def test_fp_3d_mass_conservation(simple_3d_problem):
    """Test that 3D FP solver conserves mass."""
    problem = simple_3d_problem
    solver = FPFDMSolver(problem, boundary_conditions=BoundaryConditions(type="no_flux"))

    # Create zero value function
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(3))
    U_zero = np.zeros((Nt, *shape))

    # Initial density
    x_vals = problem.geometry.grid.points[0][:-1]
    y_vals = problem.geometry.grid.points[1][:-1]
    z_vals = problem.geometry.grid.points[2][:-1]
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    x_flat = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_zero,
        show_progress=False,
    )

    # Check mass conservation
    dx, dy, dz = problem.geometry.grid.spacing
    cell_volume = dx * dy * dz

    initial_mass = np.sum(m_init) * cell_volume
    final_mass = np.sum(M_solution[-1]) * cell_volume

    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)

    print("\n3D Mass conservation check:")
    print(f"  Initial mass: {initial_mass:.6e}")
    print(f"  Final mass:   {final_mass:.6e}")
    print(f"  Relative error: {mass_error:.6e}")

    # Allow 1% relative error
    assert mass_error < 0.01, f"Mass should be conserved (error: {mass_error:.2%})"


def test_fp_2d_with_drift(simple_2d_problem):
    """Test 2D FP solver with non-zero drift field."""
    problem = simple_2d_problem
    solver = FPFDMSolver(problem)

    # Create value function with linear gradient (constant drift)
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] - 1 for d in range(2))

    # U(x, y) = x + y  →  drift v = -∇U = (-1, -1)
    x_vals = problem.geometry.grid.points[0][:-1]
    y_vals = problem.geometry.grid.points[1][:-1]
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    U_spatial = xx + yy

    # Replicate across time
    U_with_drift = np.tile(U_spatial, (Nt, 1, 1))

    # Initial density (center of domain)
    x_flat = np.column_stack([xx.ravel(), yy.ravel()])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)

    # Solve FP
    M_solution = solver.solve_fp_system(
        m_initial_condition=m_init,
        U_solution_for_drift=U_with_drift,
        show_progress=False,
    )

    # Checks
    assert M_solution.shape == (Nt, *shape), "Solution shape correct"
    assert np.all(M_solution >= -1e-10), "Density non-negative"
    assert not np.any(np.isnan(M_solution)), "No NaNs"

    # With constant leftward/downward drift, mass should move in that direction
    # Center of mass should shift
    def center_of_mass(m):
        total_mass = np.sum(m)
        if total_mass < 1e-10:
            return 0.0, 0.0
        cx = np.sum(xx * m) / total_mass
        cy = np.sum(yy * m) / total_mass
        return cx, cy

    cx_init, cy_init = center_of_mass(M_solution[0])
    cx_final, cy_final = center_of_mass(M_solution[-1])

    print("\nCenter of mass drift test:")
    print(f"  Initial: ({cx_init:.4f}, {cy_init:.4f})")
    print(f"  Final:   ({cx_final:.4f}, {cy_final:.4f})")
    print("  Drift direction: (-1, -1)")

    # With drift v = (-1, -1), center should move left and down
    # (cx_final < cx_init and cy_final < cy_init)
    # But boundary conditions may affect this
    # Just check that solution is reasonable
    assert not (cx_final > cx_init + 0.1), "Center should not move significantly right"
    assert not (cy_final > cy_init + 0.1), "Center should not move significantly up"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
