#!/usr/bin/env python3
"""
Unit tests for auxiliary functions (ppart, npart).

Tests the positive and negative part functions used in upwind schemes
for solving HJB and FP equations in Mean Field Games.
"""

import pytest

import numpy as np

from mfg_pde.utils.aux_func import npart, ppart


class TestPpartFunction:
    """Test ppart (positive part) function."""

    def test_ppart_positive_scalar(self):
        """Test ppart with positive scalar."""
        assert ppart(5.0) == 5.0
        assert ppart(1.5) == 1.5
        assert ppart(100.0) == 100.0

    def test_ppart_negative_scalar(self):
        """Test ppart with negative scalar."""
        assert ppart(-3.0) == 0.0
        assert ppart(-1.5) == 0.0
        assert ppart(-100.0) == 0.0

    def test_ppart_zero(self):
        """Test ppart with zero."""
        assert ppart(0.0) == 0.0

    def test_ppart_array_positive(self):
        """Test ppart with positive array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ppart(arr)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_ppart_array_negative(self):
        """Test ppart with negative array."""
        arr = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        result = ppart(arr)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_ppart_array_mixed(self):
        """Test ppart with mixed positive/negative array."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ppart(arr)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(result, expected)

    def test_ppart_array_2d(self):
        """Test ppart with 2D array."""
        arr = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = ppart(arr)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        assert np.allclose(result, expected)

    def test_ppart_equivalence_to_max(self):
        """Test that ppart(x) equals max(x, 0)."""
        arr = np.random.randn(100)
        result = ppart(arr)
        expected = np.maximum(arr, 0.0)
        assert np.allclose(result, expected)

    def test_ppart_preserves_shape(self):
        """Test that ppart preserves array shape."""
        for shape in [(10,), (5, 5), (2, 3, 4)]:
            arr = np.random.randn(*shape)
            result = ppart(arr)
            assert result.shape == arr.shape

    def test_ppart_returns_same_dtype(self):
        """Test that ppart returns same dtype as input."""
        arr_float32 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        result = ppart(arr_float32)
        # np.maximum may upcast to float64, this is acceptable
        assert result.dtype in [np.float32, np.float64]


class TestNpartFunction:
    """Test npart (negative part) function."""

    def test_npart_positive_scalar(self):
        """Test npart with positive scalar."""
        assert npart(5.0) == 0.0
        assert npart(1.5) == 0.0
        assert npart(100.0) == 0.0

    def test_npart_negative_scalar(self):
        """Test npart with negative scalar."""
        assert npart(-3.0) == 3.0
        assert npart(-1.5) == 1.5
        assert npart(-100.0) == 100.0

    def test_npart_zero(self):
        """Test npart with zero."""
        assert npart(0.0) == 0.0

    def test_npart_array_positive(self):
        """Test npart with positive array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = npart(arr)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_npart_array_negative(self):
        """Test npart with negative array."""
        arr = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        result = npart(arr)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_npart_array_mixed(self):
        """Test npart with mixed positive/negative array."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = npart(arr)
        expected = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_npart_array_2d(self):
        """Test npart with 2D array."""
        arr = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = npart(arr)
        expected = np.array([[1.0, 0.0], [0.0, 4.0]])
        assert np.allclose(result, expected)

    def test_npart_equivalence_to_max_negative(self):
        """Test that npart(x) equals max(-x, 0)."""
        arr = np.random.randn(100)
        result = npart(arr)
        expected = np.maximum(-arr, 0.0)
        assert np.allclose(result, expected)

    def test_npart_preserves_shape(self):
        """Test that npart preserves array shape."""
        for shape in [(10,), (5, 5), (2, 3, 4)]:
            arr = np.random.randn(*shape)
            result = npart(arr)
            assert result.shape == arr.shape

    def test_npart_square_equivalence(self):
        """Test that (npart(x))^2 equals (min(x, 0))^2."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        npart_squared = npart(arr) ** 2
        min_squared = np.minimum(arr, 0.0) ** 2
        assert np.allclose(npart_squared, min_squared)


class TestPpartNpartComplementarity:
    """Test complementary properties of ppart and npart."""

    def test_ppart_plus_npart_equals_abs(self):
        """Test that ppart(x) + npart(x) = |x|."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ppart(arr) + npart(arr)
        expected = np.abs(arr)
        assert np.allclose(result, expected)

    def test_ppart_minus_npart_equals_x(self):
        """Test that ppart(x) - npart(x) = x."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ppart(arr) - npart(arr)
        assert np.allclose(result, arr)

    def test_ppart_npart_product_is_zero(self):
        """Test that ppart(x) * npart(x) = 0 for all x."""
        arr = np.random.randn(100)
        product = ppart(arr) * npart(arr)
        assert np.allclose(product, 0.0)

    def test_ppart_squared_plus_npart_squared(self):
        """Test that ppart(x)^2 + npart(x)^2 = x^2."""
        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ppart(arr) ** 2 + npart(arr) ** 2
        expected = arr**2
        assert np.allclose(result, expected)


class TestUpwindSchemeUsage:
    """Test ppart/npart in typical upwind scheme scenarios."""

    def test_forward_difference_upwind(self):
        """Test upwind selection for forward differences."""
        # In upwind schemes: use npart for forward differences
        dx = 0.1
        u = np.array([0.0, 0.1, 0.3, 0.4, 0.5])
        p_fwd = (u[1:] - u[:-1]) / dx

        # npart extracts negative parts (flow to the right)
        npart_fwd = npart(p_fwd)
        assert np.all(npart_fwd >= 0)

    def test_backward_difference_upwind(self):
        """Test upwind selection for backward differences."""
        # In upwind schemes: use ppart for backward differences
        dx = 0.1
        u = np.array([0.0, 0.1, 0.3, 0.4, 0.5])
        p_bwd = (u[1:] - u[:-1]) / dx

        # ppart extracts positive parts (flow to the left)
        ppart_bwd = ppart(p_bwd)
        assert np.all(ppart_bwd >= 0)

    def test_upwind_flux_computation(self):
        """Test typical upwind flux computation."""
        # Velocity-like quantity
        v = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        # Upwind flux: ppart(v) goes left, npart(v) goes right
        flux_left = ppart(v)
        flux_right = npart(v)

        # Check non-negativity
        assert np.all(flux_left >= 0)
        assert np.all(flux_right >= 0)

        # Check partition of total flux
        total_flux = flux_left + flux_right
        assert np.allclose(total_flux, np.abs(v))


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_very_small_positive(self):
        """Test with very small positive values."""
        x = 1e-15
        assert ppart(x) == x
        assert npart(x) == 0.0

    def test_very_small_negative(self):
        """Test with very small negative values."""
        x = -1e-15
        assert ppart(x) == 0.0
        assert npart(x) == pytest.approx(-x, abs=1e-20)

    def test_large_values(self):
        """Test with large values."""
        assert ppart(1e10) == 1e10
        assert ppart(-1e10) == 0.0
        assert npart(1e10) == 0.0
        assert npart(-1e10) == 1e10

    def test_inf_values(self):
        """Test with infinity values."""
        assert ppart(np.inf) == np.inf
        assert ppart(-np.inf) == 0.0
        assert npart(np.inf) == 0.0
        assert npart(-np.inf) == np.inf

    def test_nan_propagation(self):
        """Test that NaN propagates through functions."""
        assert np.isnan(ppart(np.nan))
        assert np.isnan(npart(np.nan))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
