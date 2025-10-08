#!/usr/bin/env python3
"""
Test script for experimental Rust extensions.

Run after building with: maturin develop

Usage:
    cd mfg_pde_rust_experimental
    maturin develop
    python test_rust.py
"""

import numpy as np

try:
    import mfg_pde_rust_experimental as rust

    RUST_AVAILABLE = True
except ImportError as e:
    print(f"Rust extension not available: {e}")
    print("Build first with: maturin develop")
    RUST_AVAILABLE = False
    exit(1)


def test_hello_rust():
    """Test basic Rust-Python communication."""
    result = rust.hello_rust()
    print(f"✓ hello_rust(): {result}")
    assert result == "Hello from Rust! PyO3 is working."


def test_add():
    """Test basic arithmetic in Rust."""
    result = rust.add(2.0, 3.0)
    print(f"✓ add(2.0, 3.0) = {result}")
    assert result == 5.0


def test_sum_array():
    """Test NumPy array input."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rust.sum_array(arr)
    expected = np.sum(arr)
    print(f"✓ sum_array({arr}) = {result} (expected {expected})")
    assert np.isclose(result, expected)


def test_mean_array():
    """Test NumPy array processing."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rust.mean_array(arr)
    expected = np.mean(arr)
    print(f"✓ mean_array({arr}) = {result} (expected {expected})")
    assert np.isclose(result, expected)


def test_square_array():
    """Test NumPy array output."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = rust.square_array(arr)
    expected = arr**2
    print(f"✓ square_array({arr}) = {result}")
    print(f"  Expected: {expected}")
    assert np.allclose(result, expected)


def test_weno5_smoothness():
    """Test WENO5 smoothness indicators (real numerical kernel)."""
    # Test stencil
    u = np.array([1.0, 2.0, 1.5, 3.0, 2.5])

    # Rust implementation
    beta_rust = rust.weno5_smoothness(u)

    # Python reference implementation (from mfg_pde)
    def weno5_smoothness_python(u):
        beta_0 = (13 / 12) * (u[0] - 2 * u[1] + u[2]) ** 2 + (1 / 4) * (u[0] - 4 * u[1] + 3 * u[2]) ** 2
        beta_1 = (13 / 12) * (u[1] - 2 * u[2] + u[3]) ** 2 + (1 / 4) * (u[1] - u[3]) ** 2
        beta_2 = (13 / 12) * (u[2] - 2 * u[3] + u[4]) ** 2 + (1 / 4) * (3 * u[2] - 4 * u[3] + u[4]) ** 2
        return np.array([beta_0, beta_1, beta_2])

    beta_python = weno5_smoothness_python(u)

    print(f"✓ weno5_smoothness({u})")
    print(f"  Rust:   {beta_rust}")
    print(f"  Python: {beta_python}")
    print(f"  Max error: {np.max(np.abs(beta_rust - beta_python))}")

    assert np.allclose(beta_rust, beta_python), "Rust and Python results should match!"


def test_weno5_error_handling():
    """Test WENO5 error handling for wrong stencil size."""
    arr = np.array([1.0, 2.0, 3.0])  # Only 3 points, not 5

    try:
        rust.weno5_smoothness(arr)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"✓ weno5_smoothness error handling: {e}")
        if "5 points" not in str(e):
            raise AssertionError(f"Expected '5 points' in error message, got: {e}") from None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Rust Experimental Extensions")
    print("=" * 60)

    if not RUST_AVAILABLE:
        exit(1)

    print("\n1. Basic Rust-Python communication:")
    test_hello_rust()

    print("\n2. Basic arithmetic:")
    test_add()

    print("\n3. NumPy array input:")
    test_sum_array()
    test_mean_array()

    print("\n4. NumPy array output:")
    test_square_array()

    print("\n5. Real numerical kernel (WENO5):")
    test_weno5_smoothness()
    test_weno5_error_handling()

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nNext steps for learning Rust:")
    print("  1. Read src/lib.rs to understand PyO3 basics")
    print("  2. Modify weno5_smoothness() to experiment")
    print("  3. Run 'cargo test' to run Rust unit tests")
    print("  4. Profile with: python -m timeit 'import mfg_pde_rust_experimental as r; r.weno5_smoothness(...)'")
    print("  5. Compare with Python baseline from benchmarks/baseline/")
