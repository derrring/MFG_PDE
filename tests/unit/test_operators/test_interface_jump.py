"""
Unit tests for InterfaceJumpOperator, including nD support.

Tests value and gradient jumps across level set interfaces in 1D and 2D.

Created: 2026-02-06 (Issue #605 Phase 2.2)
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc
from mfg_pde.operators.differential.interface_jump import InterfaceJumpOperator


class TestInterfaceJumpOperator1D:
    """Test 1D jump operator (existing functionality)."""

    @pytest.mark.unit
    def test_1d_gradient_jump(self):
        """Test gradient jump for piecewise-linear temperature field."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        phi = x - 0.5
        T = np.where(x < 0.5, x, 0.5)

        jump_op = InterfaceJumpOperator(grid, phi, offset_distance=2 * dx)
        grad_jump = jump_op.compute_jump(T, quantity="gradient")

        idx_jump = np.argmax(np.abs(grad_jump))
        # grad_left=1, grad_right=0 -> jump = -1
        assert np.abs(grad_jump[idx_jump] - (-1.0)) < 0.15

    @pytest.mark.unit
    def test_1d_value_jump(self):
        """Test value jump for step function."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        phi = x - 0.5
        f = np.where(x < 0.5, 1.0, 2.0)

        jump_op = InterfaceJumpOperator(grid, phi, offset_distance=2 * dx)
        value_jump = jump_op.compute_jump(f, quantity="value")

        idx_jump = np.argmax(np.abs(value_jump))
        assert np.abs(value_jump[idx_jump] - 1.0) < 0.2


class TestInterfaceJumpOperator2D:
    """Test 2D jump operator (new nD support)."""

    @pytest.mark.unit
    def test_2d_value_jump_step_function(self):
        """Test value jump across a circular interface with step field."""
        Nx, Ny = 60, 60
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx=[Nx, Ny],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circular interface at radius 0.3
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Step function: 1 inside, 2 outside
        f = np.where(phi < 0, 1.0, 2.0)

        jump_op = InterfaceJumpOperator(grid, phi, offset_distance=2 * dx)
        value_jump = jump_op.compute_jump(f, quantity="value")

        # Jump should be ~ 1.0 at interface points
        interface_mask = jump_op.interface_mask
        jumps_at_interface = value_jump[interface_mask]

        if len(jumps_at_interface) > 0:
            mean_jump = np.mean(np.abs(jumps_at_interface))
            assert mean_jump > 0.5, f"Expected jump ~1.0, got mean |jump|={mean_jump:.3f}"

    @pytest.mark.unit
    def test_2d_gradient_jump_smooth_field(self):
        """Test gradient jump for a smooth field is bounded.

        For a smooth field, the gradient jump is O(2*offset * |d^2f/dn^2|),
        not zero. For sin(2pi*x)*cos(2pi*y) with offset=3*dx, the expected
        magnitude is ~4.0 due to the (2pi)^2 second-derivative curvature.
        """
        Nx, Ny = 60, 60
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx=[Nx, Ny],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circular interface
        phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3

        # Smooth field with continuous gradient everywhere
        f = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

        jump_op = InterfaceJumpOperator(grid, phi, offset_distance=3 * dx)
        grad_jump = jump_op.compute_jump(f, quantity="gradient")

        # For a smooth field, gradient jump is O(2*offset * max|d^2f/dn^2|)
        interface_mask = jump_op.interface_mask
        jumps_at_interface = grad_jump[interface_mask]

        if len(jumps_at_interface) > 0:
            max_jump = np.max(np.abs(jumps_at_interface))
            # Expected: ~6*dx*(2pi)^2 ≈ 4.0 for this test function
            assert max_jump < 5.0, f"Gradient jump too large for smooth field: {max_jump:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
