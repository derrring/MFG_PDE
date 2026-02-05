#!/usr/bin/env python3
"""
Unit tests for geometry-agnostic IC/BC validation (Issue #682).

Tests that _get_validation_points() and _get_expected_ic_shape() work
correctly across different geometry types, using lightweight mock
geometries that implement GeometryProtocol.

Tests:
- _get_validation_points(): sample extraction from various geometry shapes
- _get_expected_ic_shape(): shape inference for array validation
- Callable validation: works with structured grids, unstructured meshes, networks
- Array validation: shape checking for different geometry types
"""

import pytest

import numpy as np

from mfg_pde.utils.validation.components import (
    _get_expected_ic_shape,
    _get_validation_points,
    validate_m_initial,
    validate_u_final,
)

# ===========================================================================
# Mock geometries implementing GeometryProtocol (lightweight, no heavy deps)
# ===========================================================================


class _MockCartesianGrid:
    """Mock 1D structured grid (like TensorProductGrid)."""

    def __init__(self, n_points: int = 21):
        self._n = n_points
        self._points = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int, ...]:
        return (self._n,)

    def get_collocation_points(self) -> np.ndarray:
        return self._points  # (N, 1)

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockCartesianGrid2D:
    """Mock 2D structured grid."""

    def __init__(self, nx: int = 11, ny: int = 11):
        self._nx = nx
        self._ny = ny
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        self._points = np.column_stack([xx.ravel(), yy.ravel()])

    @property
    def dimension(self) -> int:
        return 2

    @property
    def num_spatial_points(self) -> int:
        return self._nx * self._ny

    def get_grid_shape(self) -> tuple[int, ...]:
        return (self._nx, self._ny)

    def get_collocation_points(self) -> np.ndarray:
        return self._points  # (N, 2)

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockUnstructuredMesh2D:
    """Mock 2D unstructured mesh (like Mesh2D)."""

    def __init__(self, n_vertices: int = 50):
        self._n = n_vertices
        rng = np.random.default_rng(42)
        self._points = rng.uniform(0.0, 1.0, size=(n_vertices, 2))

    @property
    def dimension(self) -> int:
        return 2

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int]:
        return (self._n,)  # Unstructured: (N,)

    def get_collocation_points(self) -> np.ndarray:
        return self._points  # (N, 2)

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockNetworkGeometry:
    """Mock network with spatial embedding (like NetworkGeometry)."""

    def __init__(self, n_nodes: int = 10, embed_dim: int = 2):
        self._n = n_nodes
        self._dim = embed_dim
        rng = np.random.default_rng(123)
        self._positions = rng.uniform(0.0, 1.0, size=(n_nodes, embed_dim))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int]:
        return (self._n,)

    def get_collocation_points(self) -> np.ndarray:
        return self._positions  # (N, d)

    def get_spatial_grid(self) -> np.ndarray:
        return self._positions


class _MockImplicitDomain:
    """Mock implicit domain (like Hyperrectangle) with sampled points."""

    def __init__(self, n_samples: int = 100, dim: int = 3):
        self._n = n_samples
        self._dim = dim
        rng = np.random.default_rng(99)
        self._points = rng.uniform(0.0, 1.0, size=(n_samples, dim))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int]:
        return (self._n,)

    def get_collocation_points(self) -> np.ndarray:
        return self._points  # (N, d)

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockGeometryNullShape:
    """Mock geometry where get_grid_shape() returns None (base Geometry behavior)."""

    def __init__(self, n_points: int = 30):
        self._n = n_points
        self._points = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self):
        return None  # Base Geometry class can return None

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


# ===========================================================================
# _get_validation_points() tests
# ===========================================================================


@pytest.mark.unit
def test_validation_points_cartesian_1d():
    """Sample points from 1D structured grid."""
    geom = _MockCartesianGrid(n_points=21)
    points = _get_validation_points(geom, n_samples=3)

    assert points.shape == (3, 1)
    # Points should be interior (not first/last)
    assert points[0, 0] > 0.0
    assert points[-1, 0] < 1.0


@pytest.mark.unit
def test_validation_points_cartesian_2d():
    """Sample points from 2D structured grid."""
    geom = _MockCartesianGrid2D(nx=11, ny=11)
    points = _get_validation_points(geom, n_samples=3)

    assert points.shape == (3, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)


@pytest.mark.unit
def test_validation_points_unstructured_mesh():
    """Sample points from unstructured mesh."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    points = _get_validation_points(geom, n_samples=3)

    assert points.shape == (3, 2)


@pytest.mark.unit
def test_validation_points_network():
    """Sample points from spatially-embedded network."""
    geom = _MockNetworkGeometry(n_nodes=10, embed_dim=2)
    points = _get_validation_points(geom, n_samples=3)

    assert points.shape == (3, 2)


@pytest.mark.unit
def test_validation_points_implicit_3d():
    """Sample points from 3D implicit domain."""
    geom = _MockImplicitDomain(n_samples=100, dim=3)
    points = _get_validation_points(geom, n_samples=3)

    assert points.shape == (3, 3)


@pytest.mark.unit
def test_validation_points_fewer_than_requested():
    """When geometry has fewer points than requested, return all available."""
    geom = _MockCartesianGrid(n_points=2)
    points = _get_validation_points(geom, n_samples=5)

    assert points.shape == (2, 1)


@pytest.mark.unit
def test_validation_points_single_sample():
    """Single sample point extraction."""
    geom = _MockCartesianGrid2D(nx=11, ny=11)
    points = _get_validation_points(geom, n_samples=1)

    assert points.shape == (1, 2)


# ===========================================================================
# _get_expected_ic_shape() tests
# ===========================================================================


@pytest.mark.unit
def test_expected_shape_cartesian_1d():
    """1D structured grid returns (N,) shape."""
    geom = _MockCartesianGrid(n_points=21)
    assert _get_expected_ic_shape(geom) == (21,)


@pytest.mark.unit
def test_expected_shape_cartesian_2d():
    """2D structured grid returns (Nx, Ny) shape."""
    geom = _MockCartesianGrid2D(nx=11, ny=13)
    assert _get_expected_ic_shape(geom) == (11, 13)


@pytest.mark.unit
def test_expected_shape_unstructured():
    """Unstructured mesh returns (N,) shape."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    assert _get_expected_ic_shape(geom) == (50,)


@pytest.mark.unit
def test_expected_shape_network():
    """Network geometry returns (N,) shape."""
    geom = _MockNetworkGeometry(n_nodes=10)
    assert _get_expected_ic_shape(geom) == (10,)


@pytest.mark.unit
def test_expected_shape_null_fallback():
    """Geometry returning None from get_grid_shape() falls back to (num_spatial_points,)."""
    geom = _MockGeometryNullShape(n_points=30)
    assert _get_expected_ic_shape(geom) == (30,)


# ===========================================================================
# Callable validation with different geometry types
# ===========================================================================


@pytest.mark.unit
def test_callable_validation_cartesian_1d():
    """Callable validation works with 1D structured grid."""
    geom = _MockCartesianGrid(n_points=21)

    def m_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    result = validate_m_initial(m_initial, geom)
    assert result.is_valid
    assert result.context.get("geometry_type") == "_MockCartesianGrid"


@pytest.mark.unit
def test_callable_validation_cartesian_2d():
    """Callable validation works with 2D structured grid."""
    geom = _MockCartesianGrid2D(nx=11, ny=11)

    def m_initial(x):
        return np.exp(-(x[0] ** 2 + x[1] ** 2))

    result = validate_m_initial(m_initial, geom)
    assert result.is_valid


@pytest.mark.unit
def test_callable_validation_unstructured_mesh():
    """Callable validation works with unstructured 2D mesh."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)

    def m_initial(x):
        return np.exp(-np.sum(x**2))

    result = validate_m_initial(m_initial, geom)
    assert result.is_valid


@pytest.mark.unit
def test_callable_validation_network():
    """Callable validation works with spatially-embedded network."""
    geom = _MockNetworkGeometry(n_nodes=10, embed_dim=2)

    def m_initial(x):
        return 1.0 / (1.0 + np.sum(x**2))

    result = validate_m_initial(m_initial, geom)
    assert result.is_valid


@pytest.mark.unit
def test_callable_validation_implicit_3d():
    """Callable validation works with 3D implicit domain."""
    geom = _MockImplicitDomain(n_samples=100, dim=3)

    def m_initial(x):
        return np.exp(-np.sum(x**2))

    result = validate_m_initial(m_initial, geom)
    assert result.is_valid


@pytest.mark.unit
def test_callable_validation_u_final_2d():
    """u_final validation works with 2D grid."""
    geom = _MockCartesianGrid2D(nx=11, ny=11)

    def u_final(x):
        return x[0] ** 2 + x[1] ** 2

    result = validate_u_final(u_final, geom)
    assert result.is_valid


@pytest.mark.unit
def test_callable_returning_nan_detected():
    """NaN return detected for non-TensorProductGrid geometry."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)

    def bad_func(x):
        return float("nan")

    result = validate_m_initial(bad_func, geom)
    assert not result.is_valid
    assert any("non-finite" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_callable_crashing_detected():
    """Exception-raising callable detected for mock geometry."""
    geom = _MockNetworkGeometry(n_nodes=10)

    def always_fails(*args):
        raise RuntimeError("I always fail")

    result = validate_m_initial(always_fails, geom)
    assert not result.is_valid


# ===========================================================================
# Array validation with different geometry types
# ===========================================================================


@pytest.mark.unit
def test_array_validation_cartesian_1d():
    """Array validation with 1D grid shape."""
    geom = _MockCartesianGrid(n_points=21)
    arr = np.ones(21)

    result = validate_m_initial(arr, geom)
    assert result.is_valid


@pytest.mark.unit
def test_array_validation_cartesian_2d():
    """Array validation with 2D grid shape (Nx, Ny)."""
    geom = _MockCartesianGrid2D(nx=11, ny=13)
    arr = np.ones((11, 13))

    result = validate_m_initial(arr, geom)
    assert result.is_valid


@pytest.mark.unit
def test_array_validation_unstructured():
    """Array validation with unstructured (N,) shape."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    arr = np.ones(50)

    result = validate_m_initial(arr, geom)
    assert result.is_valid


@pytest.mark.unit
def test_array_validation_network():
    """Array validation with network (N,) shape."""
    geom = _MockNetworkGeometry(n_nodes=10)
    arr = np.ones(10) / 10

    result = validate_m_initial(arr, geom)
    assert result.is_valid


@pytest.mark.unit
def test_array_wrong_shape_unstructured():
    """Wrong array shape detected for unstructured geometry."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    arr = np.ones(30)  # Wrong: 30 != 50

    result = validate_m_initial(arr, geom)
    assert not result.is_valid
    assert any("shape" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_array_wrong_shape_2d_grid():
    """Wrong array shape detected for 2D structured grid."""
    geom = _MockCartesianGrid2D(nx=11, ny=13)
    arr = np.ones(143)  # Flat instead of (11, 13)

    result = validate_m_initial(arr, geom)
    assert not result.is_valid


@pytest.mark.unit
def test_array_nan_unstructured():
    """NaN detected in array for unstructured geometry."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    arr = np.ones(50)
    arr[10] = np.nan

    result = validate_m_initial(arr, geom)
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_array_validation_null_shape_fallback():
    """Array validation works when get_grid_shape() returns None."""
    geom = _MockGeometryNullShape(n_points=30)
    arr = np.ones(30)

    result = validate_m_initial(arr, geom)
    assert result.is_valid


@pytest.mark.unit
def test_array_wrong_shape_null_fallback():
    """Wrong shape detected when using num_spatial_points fallback."""
    geom = _MockGeometryNullShape(n_points=30)
    arr = np.ones(25)

    result = validate_m_initial(arr, geom)
    assert not result.is_valid


# ===========================================================================
# Error message quality
# ===========================================================================


@pytest.mark.unit
def test_error_message_includes_geometry_type():
    """Shape mismatch error message includes geometry type name."""
    geom = _MockUnstructuredMesh2D(n_vertices=50)
    arr = np.ones(30)

    result = validate_m_initial(arr, geom)
    assert not result.is_valid
    # Suggestion should mention the geometry type
    error_text = " ".join(str(issue) for issue in result.issues)
    assert "_MockUnstructuredMesh2D" in error_text
