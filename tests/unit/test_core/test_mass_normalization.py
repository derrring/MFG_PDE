#!/usr/bin/env python3
"""
Unit tests for mass normalization validation (Issue #683).

Tests that validate_mass_normalization() computes mass correctly for
different geometry types and emits appropriate warnings.
"""

import pytest

import numpy as np

from mfg_pde.geometry.protocol import GeometryType
from mfg_pde.utils.validation.components import validate_mass_normalization

# ===========================================================================
# Mock geometries for mass computation testing
# ===========================================================================


class _MockCartesianGrid1D:
    """1D Cartesian grid [0, 1] with N points."""

    def __init__(self, n_points: int = 101):
        self._n = n_points
        self._points = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int, ...]:
        return (self._n,)

    def get_bounds(self):
        return (np.array([0.0]), np.array([1.0]))

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockCartesianGrid2D:
    """2D Cartesian grid [0, Lx] x [0, Ly]."""

    def __init__(self, nx: int = 21, ny: int = 21, lx: float = 1.0, ly: float = 1.0):
        self._nx = nx
        self._ny = ny
        self._lx = lx
        self._ly = ly
        xs = np.linspace(0.0, lx, nx)
        ys = np.linspace(0.0, ly, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        self._points = np.column_stack([xx.ravel(), yy.ravel()])

    @property
    def dimension(self) -> int:
        return 2

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        return self._nx * self._ny

    def get_grid_shape(self) -> tuple[int, ...]:
        return (self._nx, self._ny)

    def get_bounds(self):
        return (np.array([0.0, 0.0]), np.array([self._lx, self._ly]))

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockImplicitDomain:
    """Implicit domain with known volume and compute_volume() method."""

    def __init__(self, n_samples: int = 100, dim: int = 2, volume: float = 1.0):
        self._n = n_samples
        self._dim = dim
        self._volume = volume
        rng = np.random.default_rng(42)
        self._points = rng.uniform(0.0, 1.0, size=(n_samples, dim))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.IMPLICIT

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int]:
        return (self._n,)

    def get_bounds(self):
        return (np.zeros(self._dim), np.ones(self._dim))

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        return self._volume


class _MockNetworkGeometry:
    """Network geometry (mass computation not supported)."""

    def __init__(self, n_nodes: int = 10):
        self._n = n_nodes
        rng = np.random.default_rng(99)
        self._points = rng.uniform(0.0, 1.0, size=(n_nodes, 2))

    @property
    def dimension(self) -> int:
        return 2

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.NETWORK

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int]:
        return (self._n,)

    def get_bounds(self):
        return (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


# ===========================================================================
# Cartesian grid mass computation tests
# ===========================================================================


@pytest.mark.unit
def test_mass_normalized_1d_uniform():
    """Uniform density=1 on [0,1]: trapezoidal integral = exactly 1."""
    n = 101
    geom = _MockCartesianGrid1D(n_points=n)
    values = np.ones(n)

    result = validate_mass_normalization(values, geom, tolerance=0.1)
    assert result.is_valid
    mass = result.context.get("computed_mass")
    assert mass is not None
    # Trapezoidal rule on uniform density is exact
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_unnormalized_1d_warns():
    """Unnormalized density on [0,1] emits warning."""
    n = 101
    geom = _MockCartesianGrid1D(n_points=n)
    values = np.ones(n) * 5.0  # Integral = 5.0

    result = validate_mass_normalization(values, geom, tolerance=0.1)
    assert len(result.warnings) >= 1
    assert "mass" in str(result.warnings[0]).lower()
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 5.0) < 1e-10


@pytest.mark.unit
def test_mass_normalized_2d_uniform():
    """Uniform density=1 on [0,1]^2: trapezoidal integral = exactly 1."""
    nx, ny = 21, 21
    geom = _MockCartesianGrid2D(nx=nx, ny=ny)
    values = np.ones((nx, ny))

    result = validate_mass_normalization(values, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_computation_2d_non_unit_domain():
    """Mass computation on [0,2]x[0,3] domain."""
    nx, ny = 21, 21
    geom = _MockCartesianGrid2D(nx=nx, ny=ny, lx=2.0, ly=3.0)
    # Uniform density = 1/(2*3) = 1/6 on [0,2]x[0,3] → integral = 1
    values = np.ones((nx, ny)) / 6.0

    result = validate_mass_normalization(values, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_computation_trapezoidal_1d():
    """Verify trapezoidal rule: constant density integrates exactly."""
    n = 11
    geom = _MockCartesianGrid1D(n_points=n)
    # Constant density = 2.0 on [0,1] → integral = 2.0 (exact for trapezoidal)
    values = np.ones(n) * 2.0

    result = validate_mass_normalization(values, geom, tolerance=10.0)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 2.0) < 1e-10


# ===========================================================================
# Implicit domain mass computation tests
# ===========================================================================


@pytest.mark.unit
def test_mass_implicit_domain_normalized():
    """Implicit domain with uniform density and unit volume."""
    n = 200
    geom = _MockImplicitDomain(n_samples=n, dim=2, volume=1.0)
    values = np.ones(n)  # mean(1) * volume(1) = 1

    result = validate_mass_normalization(values, geom, tolerance=0.1)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 0.01


@pytest.mark.unit
def test_mass_implicit_domain_unnormalized():
    """Implicit domain with unnormalized density warns."""
    n = 200
    geom = _MockImplicitDomain(n_samples=n, dim=2, volume=2.0)
    values = np.ones(n) * 3.0  # mean(3) * volume(2) = 6

    result = validate_mass_normalization(values, geom, tolerance=0.1)
    assert len(result.warnings) >= 1
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 6.0) < 0.01


@pytest.mark.unit
def test_mass_implicit_domain_scaled_volume():
    """Correct normalization with non-unit volume."""
    n = 200
    volume = 4.0
    geom = _MockImplicitDomain(n_samples=n, dim=2, volume=volume)
    values = np.ones(n) / volume  # mean(1/4) * volume(4) = 1

    result = validate_mass_normalization(values, geom, tolerance=0.1)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 0.01


# ===========================================================================
# Unsupported geometry types
# ===========================================================================


@pytest.mark.unit
def test_mass_network_unsupported_warns():
    """Network geometry emits warning about unsupported mass check."""
    geom = _MockNetworkGeometry(n_nodes=10)
    values = np.ones(10)

    result = validate_mass_normalization(values, geom)
    assert len(result.warnings) >= 1
    assert "not supported" in str(result.warnings[0]).lower()


# ===========================================================================
# Callable m_initial evaluation
# ===========================================================================


@pytest.mark.unit
def test_mass_callable_1d():
    """Mass computation with callable m_initial on 1D grid."""
    n = 101
    geom = _MockCartesianGrid1D(n_points=n)

    def m_initial(x):
        return 1.0  # Constant density = 1 on [0,1] → mass = 1

    result = validate_mass_normalization(m_initial, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_callable_2d():
    """Mass computation with callable m_initial on 2D grid."""
    nx, ny = 21, 21
    geom = _MockCartesianGrid2D(nx=nx, ny=ny)

    def m_initial(x):
        return 1.0  # Constant density = 1 on [0,1]^2 → mass = 1

    result = validate_mass_normalization(m_initial, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_callable_bad_signature_warns():
    """Callable with incompatible signature does not crash."""
    geom = _MockCartesianGrid1D(n_points=11)

    def bad_func(a, b, c, d):
        return a + b + c + d

    result = validate_mass_normalization(bad_func, geom)
    # Should get a warning, not crash
    assert len(result.warnings) >= 1


# ===========================================================================
# Tolerance parameter
# ===========================================================================


@pytest.mark.unit
def test_mass_tight_tolerance():
    """Tight tolerance catches small deviations."""
    n = 101
    geom = _MockCartesianGrid1D(n_points=n)
    # Trapezoidal: mass = 1.05 (exact for constant density)
    values = np.ones(n) * 1.05

    result = validate_mass_normalization(values, geom, tolerance=0.01)
    assert len(result.warnings) >= 1  # |1.05 - 1| = 0.05 > 0.01


@pytest.mark.unit
def test_mass_loose_tolerance():
    """Loose tolerance accepts larger deviations."""
    n = 101
    geom = _MockCartesianGrid1D(n_points=n)
    values = np.ones(n) * 1.05  # mass = 1.05

    result = validate_mass_normalization(values, geom, tolerance=0.5)
    assert len(result.warnings) == 0  # |1.05 - 1| = 0.05 < 0.5


# ===========================================================================
# Integration with real TensorProductGrid
# ===========================================================================


@pytest.mark.unit
def test_mass_real_tensor_grid():
    """Mass computation works with actual TensorProductGrid."""
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.geometry.boundary.conditions import no_flux_bc

    n = 51
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[n],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    values = np.ones(n)  # Constant density = 1 on [0,1] → mass = 1
    result = validate_mass_normalization(values, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.unit
def test_mass_real_tensor_grid_callable():
    """Mass computation with callable on actual TensorProductGrid."""
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.geometry.boundary.conditions import no_flux_bc

    n = 51
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[n],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    def m_initial(x):
        return 1.0  # Uniform density on [0,1]

    result = validate_mass_normalization(m_initial, geom, tolerance=0.01)
    mass = result.context.get("computed_mass")
    assert mass is not None
    assert abs(mass - 1.0) < 1e-10
