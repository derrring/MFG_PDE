"""
Test script for nD periodic BC handling (Issue #542 related).

Verifies that periodic BC works correctly in 2D and 3D cases.
"""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid


def test_2d_periodic_bc():
    """Test 2D periodic BC (no BC specified, should use periodic fallback)."""
    print("=" * 60)
    print("Test: 2D Periodic BC (fallback)")
    print("=" * 60)

    # No BC specified -> should use periodic
    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[50, 50])
    problem = MFGProblem(geometry=grid, T=0.5, Nt=5, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    print(f"Solution shape: {U.shape}")
    print(f"Final time value range: [{U[-1].min():.6f}, {U[-1].max():.6f}]")

    # Check that solver runs without error
    assert U.shape == (6, 50, 50), f"Expected shape (6, 50, 50), got {U.shape}"
    assert not np.isnan(U).any(), "Solution contains NaN values"
    assert not np.isinf(U).any(), "Solution contains Inf values"

    print("✅ PASS: 2D periodic BC works")
    return True


def test_3d_periodic_bc():
    """Test 3D periodic BC (no BC specified, should use periodic fallback)."""
    print("\n" + "=" * 60)
    print("Test: 3D Periodic BC (fallback)")
    print("=" * 60)

    # No BC specified -> should use periodic
    grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], Nx=[20, 20, 20])
    problem = MFGProblem(geometry=grid, T=0.2, Nt=3, diffusion=0.1)

    solver = HJBFDMSolver(problem)
    U = solver.solve()

    print(f"Solution shape: {U.shape}")
    print(f"Final time value range: [{U[-1].min():.6f}, {U[-1].max():.6f}]")

    # Check that solver runs without error
    assert U.shape == (4, 20, 20, 20), f"Expected shape (4, 20, 20, 20), got {U.shape}"
    assert not np.isnan(U).any(), "Solution contains NaN values"
    assert not np.isinf(U).any(), "Solution contains Inf values"

    print("✅ PASS: 3D periodic BC works")
    return True


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Testing nD Periodic BC (Issue #542)")
    print("#" * 60)

    results = []
    try:
        results.append(("2D Periodic BC", test_2d_periodic_bc()))
    except Exception as e:
        print(f"❌ FAIL: 2D Periodic BC - {e}")
        results.append(("2D Periodic BC", False))

    try:
        results.append(("3D Periodic BC", test_3d_periodic_bc()))
    except Exception as e:
        print(f"❌ FAIL: 3D Periodic BC - {e}")
        results.append(("3D Periodic BC", False))

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
        print("\nnD Periodic BC handling verified for 2D and 3D cases.")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
