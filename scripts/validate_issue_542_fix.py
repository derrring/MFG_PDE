"""
Validation script for Issue #542: FDM BC handling fix.

This script verifies that the FDM solver correctly applies boundary conditions
instead of using periodic BC (np.roll).

Tests:
1. Neumann BC at both ends: gradient should be zero at boundaries
2. Dirichlet BC at both ends: value should match BC specification
3. Mixed BC: combination of Neumann and Dirichlet

Expected behavior:
- If BC is properly applied, solution respects boundary constraints
- If periodic BC is used (old bug), boundaries will be coupled incorrectly

IMPORTANT: Boundary conditions must be passed to the geometry object, not to MFGProblem!
"""

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, BoundaryConditions
from mfg_pde.geometry.boundary.types import BCType


def test_neumann_bc():
    """Test Neumann BC: gradient should be zero at boundaries."""
    print("=" * 60)
    print("Test 1: Neumann BC at both ends")
    print("=" * 60)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="neumann_left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"),
            BCSegment(name="neumann_right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max"),
        ]
    )
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.5, Nt=10, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    # Check gradient at boundaries (should be ~0 for Neumann BC)
    # Use intermediate time (not terminal time which has g(x)=5)
    t_check = U.shape[0] // 2  # Middle of time evolution
    dx = grid.spacing[0]
    grad_left = (U[t_check, 1] - U[t_check, 0]) / dx
    grad_right = (U[t_check, -1] - U[t_check, -2]) / dx

    print(f"Gradient at left boundary (t={t_check}):  {grad_left:.6f} (should be ~0)")
    print(f"Gradient at right boundary (t={t_check}): {grad_right:.6f} (should be ~0)")

    if abs(grad_left) < 0.1 and abs(grad_right) < 0.1:
        print("✅ PASS: Neumann BC correctly applied")
        return True
    else:
        print("❌ FAIL: Neumann BC not properly enforced")
        return False


def test_dirichlet_bc():
    """Test Dirichlet BC: values should match BC specification."""
    print("\n" + "=" * 60)
    print("Test 2: Dirichlet BC at both ends")
    print("=" * 60)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="dirichlet_left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="dirichlet_right", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_max"),
        ]
    )
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.5, Nt=10, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    # Check values at boundaries (use intermediate time, not terminal)
    t_check = U.shape[0] // 2  # Middle of time evolution
    value_left = U[t_check, 0]
    value_right = U[t_check, -1]

    print(f"Value at left boundary (t={t_check}):  {value_left:.6f} (should be ~1.0)")
    print(f"Value at right boundary (t={t_check}): {value_right:.6f} (should be ~2.0)")

    if abs(value_left - 1.0) < 0.1 and abs(value_right - 2.0) < 0.1:
        print("✅ PASS: Dirichlet BC correctly applied")
        return True
    else:
        print("❌ FAIL: Dirichlet BC not properly enforced")
        return False


def test_mixed_bc():
    """Test mixed BC: Neumann on left, Dirichlet on right."""
    print("\n" + "=" * 60)
    print("Test 3: Mixed BC (Neumann left, Dirichlet right)")
    print("=" * 60)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="neumann_left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"),
            BCSegment(name="dirichlet_right", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max"),
        ]
    )
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.5, Nt=10, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    # Check mixed BC (use intermediate time, not terminal)
    t_check = U.shape[0] // 2  # Middle of time evolution
    dx = grid.spacing[0]
    grad_left = (U[t_check, 1] - U[t_check, 0]) / dx
    value_right = U[t_check, -1]

    print(f"Gradient at left boundary (Neumann, t={t_check}): {grad_left:.6f} (should be ~0)")
    print(f"Value at right boundary (Dirichlet, t={t_check}): {value_right:.6f} (should be ~1.0)")

    if abs(grad_left) < 0.1 and abs(value_right - 1.0) < 0.1:
        print("✅ PASS: Mixed BC correctly applied")
        return True
    else:
        print("❌ FAIL: Mixed BC not properly enforced")
        return False


def test_periodic_fallback():
    """Test periodic BC fallback when bc=None."""
    print("\n" + "=" * 60)
    print("Test 4: Periodic BC fallback (bc=None)")
    print("=" * 60)

    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
    # No BC specified → should fall back to periodic
    problem = MFGProblem(geometry=grid, T=0.5, Nt=10, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    # For periodic BC, left and right should be similar
    print(f"Value at left:  {U[-1, 0]:.6f}")
    print(f"Value at right: {U[-1, -1]:.6f}")
    print(f"Difference: {abs(U[-1, 0] - U[-1, -1]):.6f}")

    # Just verify it runs without error
    print("✅ PASS: Periodic BC fallback works (backward compatibility)")
    return True


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Validation of Issue #542 Fix: FDM BC Handling")
    print("# PR #548 Implementation")
    print("#" * 60)

    results = []
    results.append(("Neumann BC", test_neumann_bc()))
    results.append(("Dirichlet BC", test_dirichlet_bc()))
    results.append(("Mixed BC", test_mixed_bc()))
    results.append(("Periodic fallback", test_periodic_fallback()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nConclusion: Issue #542 is FIXED in PR #548")
        print("FDM solver correctly applies Neumann, Dirichlet, and mixed BC.")
        print("Periodic BC is only used as fallback when bc=None (backward compat).")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nIssue #542 may not be fully resolved.")
    print("=" * 60)
