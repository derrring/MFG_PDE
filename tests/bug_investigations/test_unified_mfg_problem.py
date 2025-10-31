"""
Test unified MFGProblem class with dual initialization modes.

Tests:
1. Backward compatibility with legacy 1D API
2. New n-D initialization API
3. Computational feasibility warnings
4. Cost estimation functionality
"""

import warnings

from mfg_pde.core.mfg_problem import MFGProblem


def test_1d_backward_compatibility():
    """Test that legacy 1D API works exactly as before."""
    print("\n" + "=" * 80)
    print("TEST 1: Backward Compatibility (Legacy 1D API)")
    print("=" * 80)

    # Legacy 1D initialization (exactly as old code)
    problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=100)

    # Verify all legacy attributes exist and have correct values
    assert problem.dimension == 1, f"Expected dimension=1, got {problem.dimension}"
    assert problem.Nx == 100, f"Expected Nx=100, got {problem.Nx}"
    assert problem.xmin == 0.0, f"Expected xmin=0.0, got {problem.xmin}"
    assert problem.xmax == 1.0, f"Expected xmax=1.0, got {problem.xmax}"
    assert problem.Lx == 1.0, f"Expected Lx=1.0, got {problem.Lx}"
    assert problem.Dx == 0.01, f"Expected Dx=0.01, got {problem.Dx}"
    assert problem.Nt == 100, f"Expected Nt=100, got {problem.Nt}"
    assert problem.T == 1.0, f"Expected T=1.0, got {problem.T}"
    assert problem.Dt == 0.01, f"Expected Dt=0.01, got {problem.Dt}"

    # Verify grid arrays
    assert problem.xSpace is not None, "xSpace should not be None for 1D"
    assert len(problem.xSpace) == 101, f"Expected len(xSpace)=101, got {len(problem.xSpace)}"
    assert problem.tSpace is not None
    assert len(problem.tSpace) == 101

    # Verify spatial shape
    assert problem.spatial_shape == (101,), f"Expected spatial_shape=(101,), got {problem.spatial_shape}"
    assert problem.spatial_bounds == [(0.0, 1.0)]

    # Verify initial conditions exist
    assert problem.f_potential is not None
    assert problem.u_fin is not None
    assert problem.m_init is not None
    assert problem.f_potential.shape == (101,)
    assert problem.u_fin.shape == (101,)
    assert problem.m_init.shape == (101,)

    print("  Legacy attributes: ALL PASS")
    print("  - Nx, xmin, xmax, Lx, Dx: OK")
    print("  - xSpace array (101 points): OK")
    print("  - Initial conditions (f_potential, u_fin, m_init): OK")
    print("\nVERDICT: 100% BACKWARD COMPATIBLE")


def test_1d_via_nd_api():
    """Test 1D problem via new n-D API (for consistency)."""
    print("\n" + "=" * 80)
    print("TEST 2: 1D via New n-D API")
    print("=" * 80)

    # 1D via n-D API
    problem = MFGProblem(
        spatial_bounds=[(0.0, 1.0)],
        spatial_discretization=[100],
        Nt=100,
    )

    # Should still have legacy attributes for 1D
    assert problem.dimension == 1
    assert problem.Nx == 100
    assert problem.xmin == 0.0
    assert problem.xmax == 1.0
    assert problem.Dx == 0.01
    assert problem.xSpace is not None
    assert len(problem.xSpace) == 101

    # Also has new n-D attributes
    assert problem.spatial_shape == (100,)
    assert problem.spatial_bounds == [(0.0, 1.0)]
    assert problem.spatial_discretization == [100]

    print("  1D via n-D API: PASS")
    print("  - Legacy attributes still available: OK")
    print("  - New n-D attributes: OK")
    print("\nVERDICT: CONSISTENT 1D BEHAVIOR")


def test_2d_initialization():
    """Test 2D problem initialization."""
    print("\n" + "=" * 80)
    print("TEST 3: 2D Initialization (New API)")
    print("=" * 80)

    # 2D problem
    problem = MFGProblem(
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[50, 50],
        Nt=50,
    )

    # Verify dimension
    assert problem.dimension == 2, f"Expected dimension=2, got {problem.dimension}"

    # Verify n-D attributes
    assert problem.spatial_shape == (50, 50), f"Expected (50, 50), got {problem.spatial_shape}"
    assert problem.spatial_bounds == [(0, 1), (0, 1)]
    assert problem.spatial_discretization == [50, 50]

    # Verify legacy 1D attributes are None
    assert problem.xmin is None, "xmin should be None for 2D"
    assert problem.xmax is None, "xmax should be None for 2D"
    assert problem.Nx is None, "Nx should be None for 2D"
    assert problem.Dx is None, "Dx should be None for 2D"
    assert problem.xSpace is None, "xSpace should be None for 2D"

    # Verify grid object exists
    assert problem._grid is not None, "TensorProductGrid should exist for 2D"
    assert problem._grid.dimension == 2

    # Verify time domain
    assert problem.T == 1.0
    assert problem.Nt == 50
    assert problem.Dt == 0.02

    # Verify initial conditions exist with correct shape
    assert problem.f_potential.shape == (50, 50), f"Expected (50, 50), got {problem.f_potential.shape}"
    assert problem.u_fin.shape == (50, 50)
    assert problem.m_init.shape == (50, 50)

    print("  2D initialization: PASS")
    print(f"  - Dimension: {problem.dimension}D")
    print(f"  - Spatial shape: {problem.spatial_shape}")
    print("  - Grid object (TensorProductGrid): OK")
    print(f"  - Initial conditions shape: {problem.f_potential.shape}")
    print("\nVERDICT: 2D WORKS CORRECTLY")


def test_3d_initialization():
    """Test 3D problem initialization."""
    print("\n" + "=" * 80)
    print("TEST 4: 3D Initialization (New API)")
    print("=" * 80)

    # 3D problem
    problem = MFGProblem(
        spatial_bounds=[(0, 1), (0, 1), (0, 1)],
        spatial_discretization=[30, 30, 30],
        Nt=30,
    )

    # Verify dimension
    assert problem.dimension == 3, f"Expected dimension=3, got {problem.dimension}"

    # Verify n-D attributes
    assert problem.spatial_shape == (30, 30, 30), f"Expected (30, 30, 30), got {problem.spatial_shape}"
    assert problem.spatial_bounds == [(0, 1), (0, 1), (0, 1)]
    assert problem.spatial_discretization == [30, 30, 30]

    # Verify grid object
    assert problem._grid is not None
    assert problem._grid.dimension == 3

    # Verify initial conditions shape
    assert problem.f_potential.shape == (30, 30, 30)
    assert problem.u_fin.shape == (30, 30, 30)
    assert problem.m_init.shape == (30, 30, 30)

    print("  3D initialization: PASS")
    print(f"  - Dimension: {problem.dimension}D")
    print(f"  - Spatial shape: {problem.spatial_shape}")
    print(f"  - Initial conditions shape: {problem.f_potential.shape}")
    print("\nVERDICT: 3D WORKS CORRECTLY")


def test_computational_warnings():
    """Test that computational feasibility warnings appear correctly."""
    print("\n" + "=" * 80)
    print("TEST 5: Computational Feasibility Warnings")
    print("=" * 80)

    # Test 5D (should warn - dimension > 4)
    print("\n  Creating 5D problem (should warn about high dimension)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_5d = MFGProblem(
            spatial_bounds=[(0, 1)] * 5,
            spatial_discretization=[10] * 5,
            Nt=10,
        )
        # Check that at least one warning contains HIGH DIMENSION WARNING
        # (may get multiple warnings from both TensorProductGrid and MFGProblem)
        assert len(w) >= 1, f"Expected at least 1 warning for 5D, got {len(w)}"
        has_dimension_warning = any("HIGH DIMENSION WARNING" in str(warning.message) for warning in w)
        assert has_dimension_warning, "Expected HIGH DIMENSION WARNING in warnings"
        print(f"    Dimension warning detected (total warnings: {len(w)}): OK")

    # Test large grid (should warn - too many points)
    print("\n  Creating large 2D grid (should warn about memory)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_large = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],
            spatial_discretization=[2000, 2000],  # 4M spatial points
            Nt=1000,
        )
        # Should warn about memory
        assert any("MEMORY WARNING" in str(warning.message) for warning in w), "Expected memory warning"
        print("    Memory warning detected: OK")

    # Test suppression
    print("\n  Creating 5D with suppress_warnings=True (should NOT warn)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem_suppressed = MFGProblem(
            spatial_bounds=[(0, 1)] * 5,
            spatial_discretization=[10] * 5,
            Nt=10,
            suppress_warnings=True,
        )
        # Filter out any non-related warnings
        relevant_warnings = [warning for warning in w if "HIGH DIMENSION" in str(warning.message)]
        assert len(relevant_warnings) == 0, f"Expected 0 warnings with suppression, got {len(relevant_warnings)}"
        print("    No warnings (suppressed): OK")

    print("\nVERDICT: WARNINGS WORK CORRECTLY")


def test_cost_estimation():
    """Test computational cost estimation."""
    print("\n" + "=" * 80)
    print("TEST 6: Computational Cost Estimation")
    print("=" * 80)

    # 2D problem
    problem = MFGProblem(
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[50, 50],
        Nt=50,
    )

    cost = problem.get_computational_cost_estimate()

    # Verify cost estimate structure
    assert "dimension" in cost
    assert "spatial_shape" in cost
    assert "total_spatial_points" in cost
    assert "total_points" in cost
    assert "memory_per_array_mb" in cost
    assert "estimated_memory_mb" in cost
    assert "is_feasible" in cost
    assert "warnings" in cost

    # Verify values
    assert cost["dimension"] == 2
    assert cost["spatial_shape"] == (50, 50)
    assert cost["total_spatial_points"] == 50 * 50
    assert cost["total_points"] == 50 * 50 * 51
    assert cost["is_feasible"] is True  # 2D with 50x50 grid should be feasible

    print(f"  Dimension: {cost['dimension']}D")
    print(f"  Spatial shape: {cost['spatial_shape']}")
    print(f"  Total spatial points: {cost['total_spatial_points']:,}")
    print(f"  Total points (space x time): {cost['total_points']:,}")
    print(f"  Memory per array: {cost['memory_per_array_mb']:.2f} MB")
    print(f"  Estimated total memory: {cost['estimated_memory_mb']:.2f} MB")
    print(f"  Is feasible: {cost['is_feasible']}")
    print(f"  Warnings: {cost['warnings'] if cost['warnings'] else 'None'}")

    print("\nVERDICT: COST ESTIMATION WORKS")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 80)
    print("TEST 7: Error Handling")
    print("=" * 80)

    # Test ambiguous initialization
    print("  Testing ambiguous initialization (both Nx and spatial_bounds)...")
    try:
        problem = MFGProblem(
            Nx=100,
            spatial_bounds=[(0, 1)],
            spatial_discretization=[100],
        )
        assert False, "Should have raised ValueError for ambiguous initialization"
    except ValueError as e:
        assert "Ambiguous initialization" in str(e)
        print("    Correctly raised ValueError: OK")

    # Test mismatched dimensions
    print("  Testing mismatched dimensions...")
    try:
        problem = MFGProblem(
            spatial_bounds=[(0, 1), (0, 1)],  # 2D
            spatial_discretization=[100],  # 1D
        )
        assert False, "Should have raised ValueError for mismatched dimensions"
    except ValueError as e:
        assert "must have 2 elements" in str(e)
        print("    Correctly raised ValueError: OK")

    print("\nVERDICT: ERROR HANDLING WORKS")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("UNIFIED MFGProblem CLASS TEST SUITE")
    print("=" * 80)
    print("\nTesting dual initialization modes:")
    print("  1. Legacy 1D API (backward compatible)")
    print("  2. New n-D API (dimension-agnostic)")

    tests = [
        test_1d_backward_compatibility,
        test_1d_via_nd_api,
        test_2d_initialization,
        test_3d_initialization,
        test_computational_warnings,
        test_cost_estimation,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed == 0:
        print("\n  ALL TESTS PASSED!")
        print("  - Backward compatibility: 100%")
        print("  - New n-D API: Working")
        print("  - Warnings and cost estimation: Working")
    else:
        print(f"\n  {failed} TEST(S) FAILED")

    print("=" * 80)


if __name__ == "__main__":
    main()
