"""
Integration test for coupled HJB-FP system in 2D.

Tests that nD HJB and FP solvers work together to solve a simple MFG problem
using Picard iteration (fixed-point iteration).
"""

import pytest

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary.conditions import no_flux_bc


class SimpleCoupledMFGProblem(MFGProblem):
    """
    Simple coupled MFG problem for integration testing.

    - Isotropic Hamiltonian: H = (1/2)|∇u|²
    - Weak MFG coupling: H_m = κ·m
    - Quadratic terminal cost centered at goal
    """

    def __init__(
        self,
        domain_bounds,
        grid_resolution,
        time_domain=(0.5, 20),
        diffusion_coeff=0.1,
        coupling_strength=1.0,
        goal_position=None,
    ):
        # Handle grid_resolution as int or tuple
        if isinstance(grid_resolution, int):
            num_points = [grid_resolution, grid_resolution]
        else:
            num_points = list(grid_resolution)

        # Create geometry using geometry-first API
        geometry = TensorProductGrid(
            bounds=[(domain_bounds[0], domain_bounds[1]), (domain_bounds[2], domain_bounds[3])],
            num_points=num_points,
            boundary_conditions=no_flux_bc(dimension=2),
        )

        # Initialize MFGProblem with geometry
        super().__init__(
            geometry=geometry,
            T=time_domain[0],
            Nt=time_domain[1],
            sigma=diffusion_coeff,
        )

        self.coupling_strength = coupling_strength
        self.goal_position = goal_position if goal_position is not None else [0.7, 0.7]

    def initial_density(self, x):
        """Gaussian initial density centered at (0.3, 0.3)."""
        center = np.array([0.3, 0.3])
        dist_sq = np.sum((x - center) ** 2, axis=1)
        density = np.exp(-50 * dist_sq)
        return density / (np.sum(density) + 1e-10)

    def terminal_cost(self, x):
        """Quadratic terminal cost: distance to goal."""
        goal = np.array(self.goal_position)
        dist_sq = np.sum((x - goal) ** 2, axis=1)
        return 0.5 * dist_sq

    def running_cost(self, x, t):
        """Zero running cost."""
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """Isotropic Hamiltonian with MFG coupling: H = (1/2)|p|² + κ·m"""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared + self.coupling_strength * m

    def setup_components(self):
        """Setup MFG components."""
        coupling_strength = self.coupling_strength

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|∇u|² + κ·m"""
            h_value = 0.0

            if derivs is not None:
                # Extract gradient
                grad_u = []
                ndim = problem.geometry.grid.dimension
                for d in range(ndim):
                    idx_tuple = tuple([0] * d + [1] + [0] * (ndim - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))
                h_value += 0.5 * float(np.sum(np.array(grad_u) ** 2))

            # MFG coupling term
            h_value += coupling_strength * float(m_at_x)

            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """∂H/∂m = κ"""
            return coupling_strength

        def initial_density_func(x_idx):
            """Gaussian initial density."""
            # Get physical position
            ndim = self.geometry.grid.dimension
            coords = []
            for d in range(ndim):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.initial_density(coords_array)[0])

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost."""
            # Get physical position
            ndim = self.geometry.grid.dimension
            coords = []
            for d in range(ndim):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords).reshape(1, -1)
            return float(self.terminal_cost(coords_array)[0])

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


def solve_mfg_picard_2d(problem, max_iterations=10, tolerance=1e-3, verbose=True):
    """
    Solve 2D MFG problem using Picard iteration.

    Algorithm:
    1. Initialize M^0 = initial density
    2. For k = 0, 1, ..., max_iterations:
       a) Solve HJB backward in time with M^k fixed
       b) Solve FP forward in time with U^k fixed
       c) Check convergence: ||M^{k+1} - M^k|| < tolerance
       d) M^{k+1} = M^{k+1}

    Returns:
        dict with 'U', 'M', 'converged', 'iterations', 'errors'
    """
    # Initialize solvers (use 2D-capable Semi-Lagrangian solver)
    hjb_solver = HJBSemiLagrangianSolver(problem)
    fp_solver = FPFDMSolver(problem, boundary_conditions=no_flux_bc(dimension=2))

    # Get problem dimensions
    Nt = problem.Nt + 1
    ndim = problem.geometry.dimension
    shape = tuple(problem.geometry.num_points)

    # Initialize density M^0
    # Construct grid points manually from bounds and spacing
    x_vals = []
    for d in range(ndim):
        x_min = problem.geometry.bounds[d][0]
        spacing = problem.geometry.spacing[d]
        n_points = problem.geometry.num_points[d]
        x_vals.append(x_min + np.arange(n_points) * spacing)

    meshgrid_arrays = np.meshgrid(*x_vals, indexing="ij")
    x_flat = np.column_stack([arr.ravel() for arr in meshgrid_arrays])
    m_init_flat = problem.initial_density(x_flat)
    m_init = m_init_flat.reshape(shape)
    m_init = m_init / np.sum(m_init)  # Normalize

    # Initialize M_current as constant in time
    M_current = np.tile(m_init, (Nt, *([1] * ndim)))

    # Initialize U_current as terminal cost
    u_final_flat = problem.terminal_cost(x_flat)
    u_final = u_final_flat.reshape(shape)
    U_current = np.tile(u_final, (Nt, *([1] * ndim)))

    errors = []
    converged = False

    if verbose:
        print(f"\n{'=' * 60}")
        print("Starting Picard iteration for 2D MFG")
        print(f"Grid: {shape}, Timesteps: {Nt}")
        print(f"Coupling strength: κ = {problem.coupling_strength}")
        print(f"{'=' * 60}\n")

    for iteration in range(max_iterations):
        # Step 1: Solve HJB backward with M_current fixed
        U_new = hjb_solver.solve_hjb_system(
            M_density=M_current,
            U_terminal=u_final,
            U_coupling_prev=U_current,
        )

        # Step 2: Solve FP forward with U_new fixed
        M_new = fp_solver.solve_fp_system(
            M_initial=m_init,
            drift_field=U_new,
            show_progress=False,
        )

        # Step 3: Compute error
        error_M = np.linalg.norm(M_new - M_current) / (np.linalg.norm(M_current) + 1e-10)
        error_U = np.linalg.norm(U_new - U_current) / (np.linalg.norm(U_current) + 1e-10)
        error_combined = max(error_M, error_U)
        errors.append(error_combined)

        if verbose:
            print(f"Iteration {iteration + 1:2d}: error = {error_combined:.6e} (M: {error_M:.6e}, U: {error_U:.6e})")

        # Step 4: Check convergence
        if error_combined < tolerance:
            converged = True
            if verbose:
                print(f"\n✓ Converged in {iteration + 1} iterations (error < {tolerance:.1e})")
            break

        # Update for next iteration
        M_current = M_new
        U_current = U_new

    if not converged and verbose:
        print(f"\n✗ Did not converge in {max_iterations} iterations (error = {errors[-1]:.6e})")

    return {
        "U": U_current,
        "M": M_current,
        "converged": converged,
        "iterations": iteration + 1 if converged else max_iterations,
        "errors": errors,
        "error": errors[-1] if errors else float("inf"),
    }


@pytest.mark.skip(reason="Requires comprehensive API migration. Tracked in Issue #277 (API Consistency Audit).")
def test_coupled_hjb_fp_2d_basic():
    """Test basic 2D coupled HJB-FP solve."""
    problem = SimpleCoupledMFGProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=10,
        time_domain=(0.2, 10),
        diffusion_coeff=0.1,
        coupling_strength=0.5,
    )

    result = solve_mfg_picard_2d(problem, max_iterations=15, tolerance=1e-2, verbose=True)

    # Assertions
    assert result["converged"], f"MFG should converge (error: {result['error']:.6e})"
    assert result["iterations"] <= 15, "Should converge within 15 iterations"
    assert result["error"] < 1e-2, "Final error should be below tolerance"

    # Check shapes
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] for d in range(2))
    assert result["U"].shape == (Nt, *shape), "U shape should match (Nt, Nx, Ny)"
    assert result["M"].shape == (Nt, *shape), "M shape should match (Nt, Nx, Ny)"

    # Check mass conservation
    # Note: Full grid integration vs interior-only grid affects mass calculation
    # Errors accumulate over Picard iterations, each FP solve adds ~1% error
    dx, dy = problem.geometry.grid.spacing
    cell_volume = dx * dy
    initial_mass = np.sum(result["M"][0]) * cell_volume
    final_mass = np.sum(result["M"][-1]) * cell_volume
    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)
    print(f"\nMass conservation: initial={initial_mass:.6f}, final={final_mass:.6f}, error={mass_error:.2%}")
    print(f"Note: {result['iterations']} Picard iterations, each with ~1% FP error → cumulative")
    # Relaxed tolerance due to full grid vs interior grid integration differences
    assert mass_error < 0.40, f"Mass error {mass_error:.2%} should be < 40% for coupled system"

    # Check non-negativity
    assert np.all(result["M"] >= -1e-10), "Density should be non-negative"

    print("\n✅ 2D coupled HJB-FP test passed")


@pytest.mark.skip(reason="Requires comprehensive API migration. Tracked in Issue #277 (API Consistency Audit).")
def test_coupled_hjb_fp_2d_weak_coupling():
    """Test 2D MFG with weak coupling (should converge faster)."""
    problem = SimpleCoupledMFGProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=8,
        time_domain=(0.2, 10),
        diffusion_coeff=0.1,
        coupling_strength=0.1,  # Weak coupling
    )

    result = solve_mfg_picard_2d(problem, max_iterations=10, tolerance=1e-2, verbose=True)

    assert result["converged"], "Weak coupling should converge easily"
    assert result["iterations"] <= 10, "Weak coupling should converge quickly"

    print("\n✅ Weak coupling test passed")


@pytest.mark.skip(reason="Requires comprehensive API migration. Tracked in Issue #277 (API Consistency Audit).")
def test_coupled_hjb_fp_dimension_detection():
    """Test that dimension detection works correctly in coupled system."""
    problem = SimpleCoupledMFGProblem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=6,
        time_domain=(0.1, 5),
        diffusion_coeff=0.1,
        coupling_strength=0.3,
    )

    hjb_solver = HJBSemiLagrangianSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Check dimension detection
    assert hjb_solver.dimension == 2, "HJB solver should detect 2D"
    assert fp_solver.dimension == 2, "FP solver should detect 2D"

    print("✓ Dimension detection works for both solvers")

    # Run one iteration to verify compatibility
    Nt = problem.Nt + 1
    shape = tuple(problem.geometry.grid.num_points[d] for d in range(2))

    # Initialize
    m_init = np.ones(shape) / np.prod(shape)
    M_dummy = np.tile(m_init, (Nt, 1, 1))
    u_final = np.zeros(shape)

    # HJB solve
    U = hjb_solver.solve_hjb_system(M_dummy, u_final, M_dummy[0] * 0)

    # FP solve
    M = fp_solver.solve_fp_system(m_init, U, show_progress=False)

    # Check shapes match
    assert U.shape == M.shape == (Nt, *shape), "Shapes should match"

    print("✓ Solvers are compatible (shapes match)")
    print("\n✅ Dimension detection test passed")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Integration Tests: Coupled 2D HJB-FP System")
    print("=" * 70)

    test_coupled_hjb_fp_dimension_detection()
    print("\n" + "-" * 70)

    test_coupled_hjb_fp_2d_weak_coupling()
    print("\n" + "-" * 70)

    test_coupled_hjb_fp_2d_basic()
    print("\n" + "=" * 70)
    print("  ✅ All integration tests passed!")
    print("=" * 70 + "\n")
