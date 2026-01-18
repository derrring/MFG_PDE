"""
Unit tests for level set core functionality.

Tests:
- LevelSetFunction: Container for φ, normals, curvature
- LevelSetEvolver: Evolution ∂φ/∂t + V|∇φ| = 0
- TimeDependentDomain: Time series management

Created: 2026-01-18 (Issue #592)
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.level_set import (
    LevelSetEvolver,
    LevelSetFunction,
    TimeDependentDomain,
    compute_curvature,
    reinitialize,
)


class TestLevelSetFunction:
    """Test LevelSetFunction container."""

    def test_1d_creation(self):
        """Test creating 1D level set function."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]
        phi = x - 0.5  # Interface at x=0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        assert ls.dimension == 1
        assert ls.phi.shape == (101,)
        assert ls.is_signed_distance is True

    def test_2d_creation(self):
        """Test creating 2D level set function."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[50, 50])
        X, Y = grid.meshgrid()
        phi = np.sqrt(X**2 + Y**2) - 0.3

        ls = LevelSetFunction(phi, grid)

        assert ls.dimension == 2
        assert ls.phi.shape == (51, 51)

    def test_interface_mask(self):
        """Test interface mask computation."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]
        dx = grid.spacing[0]
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid)
        mask = ls.interface_mask(width=2 * dx)

        # Should have points near x=0.5
        assert np.sum(mask) > 0
        assert np.sum(mask) < len(x)  # Not all points

    def test_normal_field_1d(self):
        """Test normal field computation in 1D."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        normals = ls.get_normal()

        assert normals.shape == (1, 101)
        # For φ = x - c, ∇φ = 1 → n = 1 (check interior only, boundaries may differ)
        interior = np.s_[10:-10]
        assert np.allclose(np.abs(normals[:, interior]), 1.0, atol=1e-6)

    def test_normal_field_2d_circle(self):
        """Test normal field for 2D circle."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[50, 50])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle centered at (0.5, 0.5)
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        normals = ls.get_normal()

        # Check unit magnitude on interface
        interface = ls.interface_mask(width=2 * dx)
        normal_mag = np.linalg.norm(normals, axis=0)

        assert np.allclose(normal_mag[interface], 1.0, atol=1e-5)

    def test_curvature_circle(self):
        """Test curvature computation for 2D circle."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[80, 80])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle: analytical curvature κ = 1/R
        radius = 0.3
        center = np.array([0.5, 0.5])
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        kappa = ls.get_curvature()

        # Check on interface
        interface = ls.interface_mask(width=2 * dx)
        kappa_interface = kappa[interface]
        kappa_analytical = 1.0 / radius

        error = np.abs(np.mean(kappa_interface) - kappa_analytical)
        assert error < 0.2 * kappa_analytical, f"Curvature error: {error}"


class TestLevelSetEvolver:
    """Test LevelSetEvolver for ∂φ/∂t + V|∇φ| = 0."""

    def test_1d_constant_velocity(self):
        """Test 1D evolution with constant velocity."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        # Initial interface at x=0.5
        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, scheme="upwind")

        # Evolve with V=1.0 for dt=0.1
        V = 1.0
        dt = 0.1
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Interface should move to x ≈ 0.6
        idx_zero = np.argmin(np.abs(phi1))
        x_interface = x[idx_zero]
        x_expected = 0.5 + V * dt

        assert np.abs(x_interface - x_expected) < 2 * dx

    def test_cfl_adaptive_substepping(self):
        """Test CFL-adaptive substepping."""
        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, cfl_max=0.9)

        # Large velocity → should trigger substepping
        V_large = 10.0
        dt = 0.1

        # Should not crash (automatic substepping)
        phi1 = evolver.evolve_step(phi0, velocity=V_large, dt=dt)

        assert phi1.shape == phi0.shape
        assert np.isfinite(phi1).all()

    def test_2d_expansion(self):
        """Test 2D circle expansion."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[40, 40])
        X, Y = grid.meshgrid()

        # Circle
        radius = 0.2
        center = np.array([0.5, 0.5])
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        evolver = LevelSetEvolver(grid)

        # Expand with V=0.5
        V = 0.5
        dt = 0.05
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Circle should expand (more negative φ values)
        area0 = np.sum(phi0 < 0)
        area1 = np.sum(phi1 < 0)

        assert area1 > area0, "Circle should expand with positive velocity"


class TestTimeDependentDomain:
    """Test TimeDependentDomain for managing φ(t) history."""

    def test_initialization(self):
        """Test creating time-dependent domain."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid, initial_time=0.0)

        assert ls_domain.current_time == 0.0
        assert ls_domain.num_snapshots == 1
        assert np.allclose(ls_domain.current_phi, phi0)

    def test_evolution_history(self):
        """Test that history is saved correctly."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Evolve 5 steps
        V = 0.5
        dt = 0.1
        n_steps = 5

        for _ in range(n_steps):
            ls_domain.evolve_step(V, dt, save_to_history=True)

        assert ls_domain.num_snapshots == n_steps + 1  # Initial + n_steps
        assert ls_domain.current_time == n_steps * dt

    def test_time_interpolation(self):
        """Test retrieving φ at intermediate times."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Create history
        for _ in range(3):
            ls_domain.evolve_step(0.5, 0.1)

        # Get φ at t=0.15 (between snapshots)
        phi_mid = ls_domain.get_phi_at_time(0.15, interpolate=True)

        assert phi_mid.shape == phi0.shape
        assert np.isfinite(phi_mid).all()

    def test_history_clearing(self):
        """Test clearing old history."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Create history
        for _ in range(10):
            ls_domain.evolve_step(0.5, 0.1)

        snapshots_before = ls_domain.num_snapshots

        # Clear before t=0.5
        ls_domain.clear_history_before(0.5)

        assert ls_domain.num_snapshots < snapshots_before

    def test_level_set_function_wrapper(self):
        """Test getting LevelSetFunction from domain."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[50])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid, is_signed_distance=True)

        ls = ls_domain.get_level_set_function()

        assert isinstance(ls, LevelSetFunction)
        assert ls.is_signed_distance is True


class TestCurvature:
    """Test standalone curvature computation."""

    def test_flat_interface_zero_curvature(self):
        """Test that flat interface has zero curvature."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[50, 50])
        X, _Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Flat interface: φ = x - 0.5
        phi = X - 0.5

        kappa = compute_curvature(phi, grid)

        # Curvature should be near zero everywhere
        interface = np.abs(phi) < 2 * dx
        kappa_interface = kappa[interface]

        assert np.mean(np.abs(kappa_interface)) < 0.01

    def test_sphere_curvature_3d(self):
        """Test 3D sphere curvature."""
        grid = TensorProductGrid(dimension=3, bounds=[(0, 1), (0, 1), (0, 1)], Nx=[25, 25, 25])
        X, Y, Z = grid.meshgrid()
        dx = grid.spacing[0]

        # Sphere: mean curvature H = 2/R
        radius = 0.3
        center = np.array([0.5, 0.5, 0.5])
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2) - radius

        kappa = compute_curvature(phi, grid)

        interface = np.abs(phi) < 2 * dx
        kappa_interface = kappa[interface]

        kappa_analytical = 2.0 / radius
        error = np.abs(np.mean(kappa_interface) - kappa_analytical)

        # Coarse 3D grid → larger tolerance
        assert error < 0.3 * kappa_analytical


class TestReinitialization:
    """Test reinitialization to maintain SDF property."""

    def test_maintains_interface(self):
        """Test that reinitialization preserves zero level set."""
        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[60, 60])
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Slightly distort
        phi_distorted = phi0 + 0.03 * np.sin(10 * X) * np.sin(10 * Y)

        # Reinitialize
        phi_reinit = reinitialize(phi_distorted, grid, max_iterations=15)

        # Find interface before and after
        interface_before = np.abs(phi_distorted) < dx
        interface_after = np.abs(phi_reinit) < dx

        coords_before = np.column_stack([X[interface_before], Y[interface_before]])
        coords_after = np.column_stack([X[interface_after], Y[interface_after]])

        # Check that interfaces are close
        if len(coords_before) > 0 and len(coords_after) > 0:
            # Sample check
            sample_idx = 0
            pt = coords_before[sample_idx]
            dists = np.linalg.norm(coords_after - pt, axis=1)
            min_dist = np.min(dists)

            # Should be within few grid points
            assert min_dist < 10 * dx

    def test_improves_sdf_property(self):
        """Test that reinitialization maintains or improves |∇φ| ≈ 1."""
        grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[100])
        x = grid.coordinates[0]

        # Start with true SDF, add small perturbation
        phi_true = x - 0.5
        phi_perturbed = phi_true + 0.02 * np.sin(20 * np.pi * x)

        grad_ops = grid.get_gradient_operator()
        grad_before = np.abs(grad_ops[0](phi_perturbed))
        dev_before = np.max(np.abs(grad_before - 1.0))

        # Reinitialize
        phi_reinit = reinitialize(phi_perturbed, grid, max_iterations=20)

        grad_after = np.abs(grad_ops[0](phi_reinit))
        dev_after = np.max(np.abs(grad_after - 1.0))

        # Should not make things significantly worse
        assert dev_after < 2 * dev_before


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v"])
