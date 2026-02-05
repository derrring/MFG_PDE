"""
Regression Test for Anderson Accelerator Multi-Dimensional Support

Tests that AndersonAccelerator correctly handles multi-dimensional arrays.

Issue:
    The original Anderson accelerator used `np.column_stack` which expects 1D arrays.
    When used with 2D arrays (e.g., MFG density m(x,y) on a grid), it would fail.

Fix Location:
    mfg_pde/utils/numerical/anderson_acceleration.py:112-207

Test Strategy:
    1. Test with 1D arrays (original functionality)
    2. Test with 2D arrays (bug case - MFG on 2D grid)
    3. Test with 3D arrays (extensibility)
    4. Verify output shape matches input shape
"""

import pytest

import numpy as np

from mfg_pde.utils.numerical.anderson_acceleration import AndersonAccelerator

pytestmark = pytest.mark.experimental


def test_1d_arrays():
    """Test that 1D arrays still work (original functionality)"""
    # Use damping for stability
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # Simple fixed-point iteration: x_{k+1} = 0.5 * x_k + 0.5
    # Fixed point is x* = 1.0
    x = np.array([0.0, 0.0, 0.0])  # 1D array

    for _i in range(10):
        f = 0.5 * x + 0.5
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation (not convergence rate)
    assert x.shape == (3,), f"Shape changed: expected (3,), got {x.shape}"
    print("✓ 1D arrays: PASS (shape preserved)")


def test_2d_arrays():
    """Test that 2D arrays work (Bug fix case - MFG density on grid)"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # Simple 2D fixed-point iteration
    x = np.zeros((5, 5))  # 2D array (like MFG density on 5×5 grid)

    for _i in range(10):
        f = 0.5 * x + 0.5
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation (not convergence)
    assert x.shape == (5, 5), f"Shape changed: expected (5, 5), got {x.shape}"
    print("✓ 2D arrays: PASS (Bug fix verified - no crash, shape preserved)")


def test_3d_arrays():
    """Test that 3D arrays work (Extensibility test)"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # Simple 3D fixed-point iteration
    x = np.zeros((3, 4, 5))  # 3D array

    for _i in range(10):
        f = 0.5 * x + 0.5
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation
    assert x.shape == (3, 4, 5), f"Shape changed: expected (3, 4, 5), got {x.shape}"
    print("✓ 3D arrays: PASS (Extensibility verified)")


def test_type2_2d():
    """Test Type II Anderson acceleration with 2D arrays"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # 2D fixed-point iteration with Type II method
    x = np.zeros((4, 4))

    for _i in range(10):
        f = 0.5 * x + 0.5
        x_next = accelerator.update(x, f, method="type2")
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation with Type II
    assert x.shape == (4, 4), f"Shape changed: expected (4, 4), got {x.shape}"
    print("✓ Type II with 2D arrays: PASS")


def test_realistic_mfg_shape():
    """Test with realistic MFG density shape (50×50 grid)"""
    accelerator = AndersonAccelerator(depth=5, beta=0.5)

    # Simulate MFG density on 50×50 spatial grid
    m = np.ones((50, 50)) / 2500  # Initial uniform density

    for _i in range(5):
        # Simulate FP evolution (simplified)
        m_next = 0.9 * m + 0.1 * np.ones((50, 50)) / 2500
        m = accelerator.update(m, m_next)

        assert m.shape == (50, 50), f"Shape mismatch: {m.shape} vs (50, 50)"

    # Main test: large 2D arrays work without crashes
    print("✓ Realistic MFG shape (50×50): PASS (no crash with large 2D arrays)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Anderson Accelerator Multi-Dimensional Regression Tests")
    print("=" * 80 + "\n")

    test_1d_arrays()
    test_2d_arrays()
    test_3d_arrays()
    test_type2_2d()
    test_realistic_mfg_shape()

    print("\n" + "=" * 80)
    print("All tests PASSED - Anderson multi-dimensional support verified!")
    print("=" * 80 + "\n")
