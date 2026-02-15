"""Test BC enforcement without corner conflicts.

Validates that Issue #542 fix works correctly by testing scenarios
that avoid corner ambiguity:
1. Same BC value on all boundaries (no conflict)
2. BC on opposite boundaries only (no adjacent boundaries)
3. Homogeneous BC (value=0 everywhere)
"""

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, BoundaryConditions
from mfg_pde.geometry.boundary.types import BCType


def test_same_value_all_boundaries():
    """Test: Same Dirichlet value on all 4 boundaries (no corner conflict)."""
    print("=" * 70)
    print("Test 1: Same value on all boundaries (BC=1.0 everywhere)")
    print("=" * 70)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max"),
            BCSegment(name="bottom", bc_type=BCType.DIRICHLET, value=1.0, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.DIRICHLET, value=1.0, boundary="y_max"),
        ]
    )

    grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.2, Nt=3, sigma=0.1)
    solver = HJBFDMSolver(problem, max_newton_iterations=10)
    U = solver.solve()

    t_check = 1
    left = U[t_check, 0, :].mean()
    right = U[t_check, -1, :].mean()
    bottom = U[t_check, :, 0].mean()
    top = U[t_check, :, -1].mean()

    print(f"\nBoundary means at t={t_check}:")
    print(f"  Left:   {left:.6f} (should be ~1.0)")
    print(f"  Right:  {right:.6f} (should be ~1.0)")
    print(f"  Bottom: {bottom:.6f} (should be ~1.0)")
    print(f"  Top:    {top:.6f} (should be ~1.0)")

    tol = 0.1
    all_ok = all(abs(val - 1.0) < tol for val in [left, right, bottom, top])

    if all_ok:
        print("\n✅ PASS: All boundaries correctly enforced")
        return True
    else:
        print("\n❌ FAIL: BC enforcement issue")
        return False


def test_opposite_boundaries_only():
    """Test: BC on opposite boundaries only (no adjacent boundaries, no corners)."""
    print("\n" + "=" * 70)
    print("Test 2: Opposite boundaries only (x_min, x_max)")
    print("=" * 70)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_max"),
        ]
    )

    grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.2, Nt=3, sigma=0.1)
    solver = HJBFDMSolver(problem, max_newton_iterations=10)
    U = solver.solve()

    t_check = 1
    left = U[t_check, 0, :].mean()
    right = U[t_check, -1, :].mean()

    print(f"\nBoundary means at t={t_check}:")
    print(f"  Left (x=0):  {left:.6f} (should be ~1.0)")
    print(f"  Right (x=1): {right:.6f} (should be ~2.0)")

    tol = 0.1
    left_ok = abs(left - 1.0) < tol
    right_ok = abs(right - 2.0) < tol

    if left_ok and right_ok:
        print("\n✅ PASS: Opposite boundaries correctly enforced")
        return True
    else:
        print("\n❌ FAIL: BC enforcement issue")
        return False


def test_homogeneous_bc():
    """Test: Homogeneous BC (value=0 everywhere)."""
    print("\n" + "=" * 70)
    print("Test 3: Homogeneous BC (value=0 on all boundaries)")
    print("=" * 70)

    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
            BCSegment(name="bottom", bc_type=BCType.DIRICHLET, value=0.0, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.DIRICHLET, value=0.0, boundary="y_max"),
        ]
    )

    grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=bc)
    problem = MFGProblem(geometry=grid, T=0.2, Nt=3, sigma=0.1)
    solver = HJBFDMSolver(problem, max_newton_iterations=10)
    U = solver.solve()

    t_check = 1
    left = U[t_check, 0, :].mean()
    right = U[t_check, -1, :].mean()
    bottom = U[t_check, :, 0].mean()
    top = U[t_check, :, -1].mean()

    print(f"\nBoundary means at t={t_check}:")
    print(f"  Left:   {left:.6f} (should be ~0.0)")
    print(f"  Right:  {right:.6f} (should be ~0.0)")
    print(f"  Bottom: {bottom:.6f} (should be ~0.0)")
    print(f"  Top:    {top:.6f} (should be ~0.0)")

    tol = 0.1
    all_ok = all(abs(val) < tol for val in [left, right, bottom, top])

    if all_ok:
        print("\n✅ PASS: Homogeneous BC correctly enforced")
        return True
    else:
        print("\n❌ FAIL: BC enforcement issue")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# BC Enforcement Validation (No Corner Conflicts)")
    print("# Issue #542: Verify core BC enforcement works")
    print("#" * 70)

    results = []

    try:
        results.append(("Same value all boundaries", test_same_value_all_boundaries()))
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Same value all boundaries", False))

    try:
        results.append(("Opposite boundaries only", test_opposite_boundaries_only()))
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Opposite boundaries only", False))

    try:
        results.append(("Homogeneous BC", test_homogeneous_bc()))
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Homogeneous BC", False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nConclusion: Issue #542 fix is correct.")
        print("BC enforcement works properly when corner conflicts avoided.")
        print("Corner conflicts (different values on adjacent boundaries) tracked in Issue #521.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nIssue #542 fix may have problems.")
    print("=" * 70)
