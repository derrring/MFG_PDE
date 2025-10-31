"""
Simple validation for full nD FP FDM solver.

Tests the consolidated fp_fdm.py solver directly without complex inheritance.
Validates that the unified structure works correctly after consolidation.
"""

import numpy as np


def test_fp_2d_pure_diffusion():
    """Test 1: 2D pure diffusion (no advection)."""
    print("=" * 70)
    print("  Test 1: 2D Pure Diffusion (No Advection)")
    print("=" * 70)
    print()
    print("Setup: 12×12 grid, T=0.5, 15 timesteps")
    print("Solver: Full nD FP FDM (unified fp_fdm.py)")
    print("Expected: ~0-2% mass error (diffusion conserves mass)")
    print()

    # Import after printing header (cleaner output)
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.geometry import BoundaryConditions, TensorProductGrid

    # Create 2D grid
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[12, 12],
    )

    # Create minimal problem (just for solver initialization)
    class MinimalProblem:
        def __init__(self, grid, T, Nt, sigma):
            # Create geometry with grid
            class SimpleGeometry:
                def __init__(self, grid, boundary_conditions):
                    self.grid = grid
                    self.boundary_conditions = boundary_conditions
                    self.dimension = grid.dimension

            self.geometry = SimpleGeometry(grid, BoundaryConditions(type="no_flux"))
            self.T = T
            self.Nt = Nt
            self.sigma = sigma
            self.dt = T / Nt

    problem = MinimalProblem(grid=grid, T=0.5, Nt=15, sigma=0.05)

    # Create FP solver
    fp_solver = FPFDMSolver(problem)

    # Initial density: Gaussian blob at center
    center = np.array([0.5, 0.5])
    points = grid.flatten()  # Get all grid points as (N, 2) array
    dist_sq = np.sum((points - center) ** 2, axis=1)
    m0_flat = np.exp(-100 * dist_sq)

    # Reshape to grid shape (12, 12)
    m0 = m0_flat.reshape(grid.num_points)

    # Normalize by integral
    dx, dy = grid.spacing
    dV = dx * dy
    m0 = m0 / (np.sum(m0) * dV + 1e-10)

    # Zero velocity field (pure diffusion)
    U_zero = np.zeros((problem.Nt + 1, *grid.num_points))

    print("Running FP solver (pure diffusion)...")
    M_solution = fp_solver.solve_fp_system(
        m_initial_condition=m0,
        U_solution_for_drift=U_zero,
        show_progress=True,
    )

    # Compute mass conservation
    initial_mass = np.sum(M_solution[0]) * dV
    final_mass = np.sum(M_solution[-1]) * dV
    error_percent = abs(final_mass - initial_mass) / (initial_mass + 1e-16) * 100

    print()
    print("Results:")
    print(f"  Initial mass: {initial_mass:.8f}")
    print(f"  Final mass:   {final_mass:.8f}")
    print(f"  Error:        {error_percent:+.4f}%")
    print()

    if error_percent < 2.0:
        print("  PASS: Mass conserved within 2%")
    else:
        print(f"  FAIL: Mass error {error_percent:.2f}% exceeds 2%")
    print()

    return error_percent < 2.0


def test_fp_2d_with_advection():
    """Test 2: 2D with non-zero velocity field."""
    print("=" * 70)
    print("  Test 2: 2D FP with Advection")
    print("=" * 70)
    print()
    print("Setup: 12×12 grid, T=0.3, 10 timesteps")
    print("Solver: Full nD FP FDM (unified fp_fdm.py)")
    print("Expected: ~1-10% mass error (advection-diffusion with FDM)")
    print()

    # Import after printing header
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.geometry import BoundaryConditions, TensorProductGrid

    # Create 2D grid
    grid = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        num_points=[12, 12],
    )

    # Create minimal problem
    class MinimalProblem:
        def __init__(self, grid, T, Nt, sigma):
            class SimpleGeometry:
                def __init__(self, grid, boundary_conditions):
                    self.grid = grid
                    self.boundary_conditions = boundary_conditions
                    self.dimension = grid.dimension

            self.geometry = SimpleGeometry(grid, BoundaryConditions(type="no_flux"))
            self.T = T
            self.Nt = Nt
            self.sigma = sigma
            self.dt = T / Nt

    problem = MinimalProblem(grid=grid, T=0.3, Nt=10, sigma=0.05)

    # Create FP solver
    fp_solver = FPFDMSolver(problem)

    # Initial density: Gaussian blob at (0.2, 0.2)
    start = np.array([0.2, 0.2])
    points = grid.flatten()
    dist_sq = np.sum((points - start) ** 2, axis=1)
    m0_flat = np.exp(-100 * dist_sq)
    m0 = m0_flat.reshape(grid.num_points)

    # Normalize
    dx, dy = grid.spacing
    dV = dx * dy
    m0 = m0 / (np.sum(m0) * dV + 1e-10)

    # Simple constant velocity field (move toward goal at (0.8, 0.8))
    # This creates a simple advection field
    U_velocity = np.zeros((problem.Nt + 1, *grid.num_points))
    # Create gradient field pointing toward (0.8, 0.8)
    xx, yy = np.meshgrid(np.linspace(0, 1, grid.num_points[0]), np.linspace(0, 1, grid.num_points[1]), indexing="ij")
    grad_x = -(xx - 0.8)  # Negative gradient = value function
    grad_y = -(yy - 0.8)
    for k in range(problem.Nt + 1):
        U_velocity[k] = grad_x + grad_y  # Simple value function

    print("Running FP solver (with advection)...")
    M_solution = fp_solver.solve_fp_system(
        m_initial_condition=m0,
        U_solution_for_drift=U_velocity,
        show_progress=True,
    )

    # Compute mass conservation
    initial_mass = np.sum(M_solution[0]) * dV
    final_mass = np.sum(M_solution[-1]) * dV
    error_percent = abs(final_mass - initial_mass) / (initial_mass + 1e-16) * 100

    print()
    print("Results:")
    print(f"  Initial mass: {initial_mass:.8f}")
    print(f"  Final mass:   {final_mass:.8f}")
    print(f"  Error:        {error_percent:+.4f}%")
    print()

    if error_percent < 10.0:
        print(f"  PASS: Mass error {error_percent:.2f}% is acceptable for FDM")
    else:
        print(f"  FAIL: Mass error {error_percent:.2f}% exceeds 10%")
    print()

    return error_percent < 10.0


def main():
    """Run all validation tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Full nD FP FDM Solver Validation" + " " * 21 + "║")
    print("║" + " " * 17 + "(After Consolidation)" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Test 1: Pure diffusion
    try:
        results.append(("Pure Diffusion", test_fp_2d_pure_diffusion()))
    except Exception as e:
        print(f"ERROR in Test 1: {e}")
        results.append(("Pure Diffusion", False))

    print()

    # Test 2: With advection
    try:
        results.append(("With Advection", test_fp_2d_with_advection()))
    except Exception as e:
        print(f"ERROR in Test 2: {e}")
        results.append(("With Advection", False))

    print()
    print("=" * 70)
    print("  Validation Summary")
    print("=" * 70)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
    print()

    all_passed = all(p for _, p in results)
    if all_passed:
        print("  All tests PASSED - Consolidation successful!")
    else:
        print("  Some tests FAILED - Review needed")
    print()


if __name__ == "__main__":
    main()
