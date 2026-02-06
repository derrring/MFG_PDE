"""
Unit tests for Godunov upwind update formulas.

These tests verify the core numerical building blocks for FMM and FSM.
"""

import pytest

import numpy as np

from mfg_pde.geometry.level_set.eikonal import (
    godunov_update_1d,
    godunov_update_2d,
    godunov_update_nd,
)


class TestGodunov1D:
    """Test 1D Godunov update."""

    def test_basic_update(self):
        """Test basic 1D update: T = T_min + dx/F."""
        T_new = godunov_update_1d((0.1, 0.3), dx=0.01, speed=1.0)
        expected = 0.1 + 0.01
        assert abs(T_new - expected) < 1e-10

    def test_selects_minimum_neighbor(self):
        """Test that upwind scheme selects minimum neighbor."""
        T_new_left_min = godunov_update_1d((0.1, 0.5), dx=0.01, speed=1.0)
        T_new_right_min = godunov_update_1d((0.5, 0.1), dx=0.01, speed=1.0)
        # Both should give same result (T_min = 0.1)
        assert abs(T_new_left_min - T_new_right_min) < 1e-10

    def test_handles_inf_neighbor(self):
        """Test handling of infinite (boundary) neighbors."""
        T_new = godunov_update_1d((0.1, np.inf), dx=0.01, speed=1.0)
        expected = 0.1 + 0.01
        assert abs(T_new - expected) < 1e-10

    def test_both_inf_returns_inf(self):
        """Test that both neighbors infinite returns inf."""
        T_new = godunov_update_1d((np.inf, np.inf), dx=0.01, speed=1.0)
        assert np.isinf(T_new)

    def test_non_unit_speed(self):
        """Test with non-unit speed function."""
        T_new = godunov_update_1d((0.0, np.inf), dx=0.01, speed=2.0)
        expected = 0.0 + 0.01 / 2.0
        assert abs(T_new - expected) < 1e-10


class TestGodunov2D:
    """Test 2D Godunov update."""

    def test_symmetric_case(self):
        """Test 2D symmetric case: T = a + dx/sqrt(2)."""
        a = 0.1
        dx = dy = 0.01
        T_new = godunov_update_2d((a, np.inf), (a, np.inf), dx, dy, speed=1.0)
        expected = a + dx / np.sqrt(2)
        assert abs(T_new - expected) < 1e-10

    def test_one_dimension_inf(self):
        """Test fallback to 1D when one dimension is inf."""
        T_new = godunov_update_2d((0.1, np.inf), (np.inf, np.inf), 0.01, 0.01, 1.0)
        expected = 0.1 + 0.01  # 1D update from x
        assert abs(T_new - expected) < 1e-10

    def test_causality_close_neighbors(self):
        """Test that T >= max(upwind neighbors) when they are close."""
        # When neighbors are close, the 2D quadratic is valid
        a, b = 0.1, 0.105
        T_new = godunov_update_2d((a, np.inf), (b, np.inf), 0.01, 0.01, 1.0)
        assert T_new >= max(a, b)

    def test_fallback_to_1d(self):
        """Test fallback to 1D when 2D quadratic is invalid."""
        # When neighbors are far apart, 2D quadratic may have no solution
        # In this case, we fall back to 1D update from the minimum neighbor
        a, b = 0.1, 0.5  # Very different values
        T_new = godunov_update_2d((a, np.inf), (b, np.inf), 0.01, 0.01, 1.0)
        # Should be 1D update from smaller neighbor: a + dx = 0.11
        expected = a + 0.01
        assert abs(T_new - expected) < 1e-10

    def test_anisotropic_spacing(self):
        """Test with different spacing in x and y."""
        a = 0.1
        dx, dy = 0.01, 0.02
        T_new = godunov_update_2d((a, np.inf), (a, np.inf), dx, dy, speed=1.0)
        # Should use quadratic formula
        # ((T-a)/dx)^2 + ((T-a)/dy)^2 = 1
        # (1/dx^2 + 1/dy^2)(T-a)^2 = 1
        # T = a + 1/sqrt(1/dx^2 + 1/dy^2)
        expected = a + 1.0 / np.sqrt(1 / dx**2 + 1 / dy**2)
        assert abs(T_new - expected) < 1e-10


class TestGodunovND:
    """Test n-dimensional Godunov update."""

    def test_reduces_to_1d(self):
        """Test that nD with 1 dimension reduces to 1D."""
        T_neighbors_1d = [(0.1, 0.3)]
        T_nd = godunov_update_nd(T_neighbors_1d, [0.01], speed=1.0)
        T_1d = godunov_update_1d((0.1, 0.3), dx=0.01, speed=1.0)
        assert abs(T_nd - T_1d) < 1e-10

    def test_reduces_to_2d(self):
        """Test that nD with 2 dimensions reduces to 2D."""
        T_neighbors_2d = [(0.1, np.inf), (0.1, np.inf)]
        T_nd = godunov_update_nd(T_neighbors_2d, [0.01, 0.01], speed=1.0)
        T_2d = godunov_update_2d((0.1, np.inf), (0.1, np.inf), 0.01, 0.01, 1.0)
        assert abs(T_nd - T_2d) < 1e-10

    def test_3d_symmetric(self):
        """Test 3D symmetric case: T = a + dx/sqrt(3)."""
        a = 0.1
        dx = 0.01
        T_neighbors_3d = [(a, np.inf), (a, np.inf), (a, np.inf)]
        T_new = godunov_update_nd(T_neighbors_3d, [dx, dx, dx], speed=1.0)
        expected = a + dx / np.sqrt(3)
        assert abs(T_new - expected) < 1e-10

    def test_3d_causality_close_neighbors(self):
        """Test causality in 3D with close neighbors."""
        # Close neighbors allow valid nD quadratic solution
        T_neighbors = [(0.1, np.inf), (0.102, np.inf), (0.104, np.inf)]
        T_new = godunov_update_nd(T_neighbors, [0.01, 0.01, 0.01], speed=1.0)
        assert T_new >= max(0.1, 0.102, 0.104)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
