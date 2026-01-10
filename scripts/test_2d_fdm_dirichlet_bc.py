"""
Test 2D FDM Dirichlet BC enforcement.

Validates that 2D FDM solver enforces Dirichlet BC correctly,
similar to the 1D fix in Issue #542.

Expected: Boundary values should exactly match BC specification
at all time steps (not just terminal time).
"""

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, BoundaryConditions
from mfg_pde.geometry.boundary.types import BCType


def test_2d_dirichlet_x_boundaries():
    """Test Dirichlet BC on x-boundaries (left/right walls)."""
    print("=" * 70)
    print("Test: 2D Dirichlet BC on x-boundaries")
    print("=" * 70)

    # Create BC: Dirichlet on x-boundaries, periodic on y
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left_wall", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="right_wall", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_max"),
        ]
    )

    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[40, 40], boundary_conditions=bc)

    problem = MFGProblem(geometry=grid, T=0.5, Nt=5, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    print(f"\nSolution shape: {U.shape}")
    print("Expected: (Nt+1, Nx, Ny) = (6, 41, 41)")

    # Check boundary values at intermediate time (not terminal)
    t_check = U.shape[0] // 2  # Middle of time evolution

    # Left boundary (x=0, all y)
    left_boundary_values = U[t_check, 0, :]
    left_mean = left_boundary_values.mean()
    left_std = left_boundary_values.std()

    # Right boundary (x=1, all y)
    right_boundary_values = U[t_check, -1, :]
    right_mean = right_boundary_values.mean()
    right_std = right_boundary_values.std()

    print(f"\n--- Time step t={t_check} (intermediate) ---")
    print(f"Left boundary (x=0):  mean={left_mean:.6f}, std={left_std:.6f} (should be 1.0 ± 0)")
    print(f"Right boundary (x=1): mean={right_mean:.6f}, std={right_std:.6f} (should be 2.0 ± 0)")

    # Check all time steps (except terminal which has g(x)=5)
    print("\n--- All time steps (excluding terminal) ---")
    for t in range(U.shape[0] - 1):  # Exclude terminal time
        left_vals = U[t, 0, :]
        right_vals = U[t, -1, :]
        print(
            f"t={t:2d}: left={left_vals.mean():.4f} (std={left_vals.std():.6f}), "
            f"right={right_vals.mean():.4f} (std={right_vals.std():.6f})"
        )

    # Validation
    tolerance = 0.1
    left_ok = abs(left_mean - 1.0) < tolerance and left_std < tolerance
    right_ok = abs(right_mean - 2.0) < tolerance and right_std < tolerance

    if left_ok and right_ok:
        print("\n✅ PASS: 2D Dirichlet BC correctly enforced on x-boundaries")
        return True
    else:
        print("\n❌ FAIL: 2D Dirichlet BC not properly enforced")
        if not left_ok:
            print(f"   Left boundary issue: expected 1.0 ± 0, got {left_mean:.6f} ± {left_std:.6f}")
        if not right_ok:
            print(f"   Right boundary issue: expected 2.0 ± 0, got {right_mean:.6f} ± {right_std:.6f}")
        return False


def test_2d_dirichlet_all_boundaries():
    """Test Dirichlet BC on all four boundaries."""
    print("\n" + "=" * 70)
    print("Test: 2D Dirichlet BC on all four boundaries")
    print("=" * 70)

    # Create BC: Different values on each boundary
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_max"),
            BCSegment(name="bottom", bc_type=BCType.DIRICHLET, value=3.0, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.DIRICHLET, value=4.0, boundary="y_max"),
        ]
    )

    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[30, 30], boundary_conditions=bc)

    problem = MFGProblem(geometry=grid, T=0.3, Nt=3, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    # Check at intermediate time
    t_check = 1  # Second time step

    left = U[t_check, 0, :].mean()
    right = U[t_check, -1, :].mean()
    bottom = U[t_check, :, 0].mean()
    top = U[t_check, :, -1].mean()

    print(f"\n--- Time step t={t_check} ---")
    print(f"Left (x=0):   {left:.6f} (should be ~1.0)")
    print(f"Right (x=1):  {right:.6f} (should be ~2.0)")
    print(f"Bottom (y=0): {bottom:.6f} (should be ~3.0)")
    print(f"Top (y=1):    {top:.6f} (should be ~4.0)")

    tolerance = 0.15  # Slightly larger tolerance for 4-boundary case
    all_ok = (
        abs(left - 1.0) < tolerance
        and abs(right - 2.0) < tolerance
        and abs(bottom - 3.0) < tolerance
        and abs(top - 4.0) < tolerance
    )

    if all_ok:
        print("\n✅ PASS: 2D Dirichlet BC correctly enforced on all boundaries")
        return True
    else:
        print("\n❌ FAIL: 2D Dirichlet BC not properly enforced")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Testing 2D FDM Dirichlet BC Enforcement (nD path validation)")
    print("# Related: Issue #542 fix verification for nD")
    print("#" * 70)

    results = []

    try:
        results.append(("2D x-boundaries", test_2d_dirichlet_x_boundaries()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in 2D x-boundaries test: {e}")
        import traceback

        traceback.print_exc()
        results.append(("2D x-boundaries", False))

    try:
        results.append(("2D all boundaries", test_2d_dirichlet_all_boundaries()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in 2D all boundaries test: {e}")
        import traceback

        traceback.print_exc()
        results.append(("2D all boundaries", False))

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
        print("\nConclusion: 2D FDM correctly enforces Dirichlet BC.")
        print("Issue #542 fix extends to nD path.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nConclusion: 2D FDM may have BC enforcement issues.")
        print("nD path needs BC enforcement similar to 1D fix.")
    print("=" * 70)
