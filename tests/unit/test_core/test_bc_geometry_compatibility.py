#!/usr/bin/env python3
"""
Unit tests for BC-geometry compatibility validation (Issue #679).

Tests that validate_boundary_conditions() catches:
- BC dimension vs geometry dimension mismatch
- Periodic BC on non-Cartesian geometries
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary.conditions import (
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)
from mfg_pde.geometry.boundary.types import BCSegment, BCType
from mfg_pde.geometry.protocol import GeometryType
from mfg_pde.utils.validation.components import validate_boundary_conditions

# ===========================================================================
# Mock geometries
# ===========================================================================


class _MockCartesianGrid:
    """1D Cartesian grid."""

    def __init__(self, dim: int = 1, n_points: int = 21):
        self._dim = dim
        self._n = n_points
        self._points = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        return self._n

    def get_grid_shape(self) -> tuple[int, ...]:
        return (self._n,)

    def get_bounds(self):
        return (np.zeros(self._dim), np.ones(self._dim))

    def get_collocation_points(self) -> np.ndarray:
        return self._points

    def get_spatial_grid(self) -> np.ndarray:
        return self._points


class _MockImplicitDomain:
    """2D implicit domain (non-Cartesian)."""

    def __init__(self, n_samples: int = 50, dim: int = 2):
        self._n = n_samples
        self._dim = dim
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


class _MockNetworkGeometry:
    """Network geometry."""

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
# BC dimension mismatch tests
# ===========================================================================


@pytest.mark.unit
def test_bc_dimension_matches_geometry():
    """No error when BC dimension matches geometry dimension."""
    geom = _MockCartesianGrid(dim=1)
    bc = neumann_bc(dimension=1)

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.issues) == 0


@pytest.mark.unit
def test_bc_dimension_mismatch_error():
    """Error when BC dimension != geometry dimension."""
    geom = _MockCartesianGrid(dim=1)
    bc = neumann_bc(dimension=2)  # Mismatch: 2D BC on 1D geometry

    result = validate_boundary_conditions(bc, geom)
    assert not result.is_valid
    assert any("dimension" in str(issue).lower() for issue in result.issues)


@pytest.mark.unit
def test_bc_dimension_mismatch_2d_on_1d():
    """Error when 2D BC applied to 1D geometry."""
    geom = _MockCartesianGrid(dim=1)
    bc = periodic_bc(dimension=2)

    result = validate_boundary_conditions(bc, geom)
    assert not result.is_valid


@pytest.mark.unit
def test_bc_dimension_mismatch_1d_on_2d():
    """Error when 1D BC applied to 2D geometry."""
    geom = _MockImplicitDomain(dim=2)
    bc = dirichlet_bc(value=0.0, dimension=1)

    result = validate_boundary_conditions(bc, geom)
    assert not result.is_valid


@pytest.mark.unit
def test_bc_unbound_dimension_skips_check():
    """No dimension error when BC has dimension=None (lazy binding)."""
    geom = _MockCartesianGrid(dim=1)
    bc = BoundaryConditions(
        dimension=None,
        segments=[BCSegment(name="all", bc_type=BCType.NEUMANN, value=0.0)],
    )

    result = validate_boundary_conditions(bc, geom)
    # Should not error — BC dimension is unbound (lazy)
    assert result.is_valid


# ===========================================================================
# Periodic BC on non-Cartesian geometry tests
# ===========================================================================


@pytest.mark.unit
def test_periodic_on_cartesian_ok():
    """Periodic BC on Cartesian grid: no warning."""
    geom = _MockCartesianGrid(dim=1)
    bc = periodic_bc(dimension=1)

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.warnings) == 0


@pytest.mark.unit
def test_periodic_on_implicit_warns():
    """Periodic BC on implicit domain: warning."""
    geom = _MockImplicitDomain(dim=2)
    bc = periodic_bc(dimension=2)

    result = validate_boundary_conditions(bc, geom)
    # Warning, not error (is_valid stays True)
    assert result.is_valid
    assert len(result.warnings) >= 1
    assert "periodic" in str(result.warnings[0]).lower()


@pytest.mark.unit
def test_periodic_on_network_warns():
    """Periodic BC on network geometry: warning."""
    geom = _MockNetworkGeometry(n_nodes=10)
    bc = periodic_bc(dimension=2)

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.warnings) >= 1
    assert "cartesian" in str(result.warnings[0]).lower()


@pytest.mark.unit
def test_periodic_segment_on_implicit_warns():
    """Periodic BC in a segment (not default) on implicit domain: warning."""
    geom = _MockImplicitDomain(dim=2)
    bc = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="periodic_x", bc_type=BCType.PERIODIC, value=0.0)],
        default_bc=BCType.NEUMANN,
    )

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.warnings) >= 1


@pytest.mark.unit
def test_non_periodic_on_implicit_ok():
    """Non-periodic BC on implicit domain: no warning."""
    geom = _MockImplicitDomain(dim=2)
    bc = neumann_bc(dimension=2)

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.warnings) == 0


@pytest.mark.unit
def test_no_flux_on_network_ok():
    """No-flux BC on network geometry: no periodic warning."""
    geom = _MockNetworkGeometry(n_nodes=10)
    bc = no_flux_bc(dimension=2)

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid
    assert len(result.warnings) == 0


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
def test_none_bc_skips_validation():
    """None boundary conditions: no validation performed."""
    geom = _MockCartesianGrid(dim=1)

    result = validate_boundary_conditions(None, geom)
    assert result.is_valid
    assert len(result.issues) == 0


@pytest.mark.unit
def test_dimension_mismatch_and_periodic_both_reported():
    """Both dimension mismatch (error) and periodic warning reported."""
    geom = _MockImplicitDomain(dim=2)
    bc = periodic_bc(dimension=3)  # Wrong dimension AND periodic on implicit

    result = validate_boundary_conditions(bc, geom)
    assert not result.is_valid  # Dimension mismatch is error
    # Should have both a dimension error and a periodic warning
    assert len(result.issues) >= 2


@pytest.mark.unit
def test_mixed_bc_with_periodic_segment():
    """Mixed BC where only one segment is periodic on non-Cartesian: warns."""
    geom = _MockImplicitDomain(dim=2)
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NEUMANN, value=0.0, boundary="left"),
            BCSegment(name="wrap", bc_type=BCType.PERIODIC, value=0.0, boundary="right"),
        ],
        default_bc=BCType.NEUMANN,
    )

    result = validate_boundary_conditions(bc, geom)
    assert result.is_valid  # Only warning, not error
    assert len(result.warnings) >= 1


# ===========================================================================
# Integration: validate_components wires in BC check
# ===========================================================================


@pytest.mark.unit
def test_validate_components_catches_bc_dimension_mismatch():
    """validate_components() surfaces BC dimension mismatch."""
    from unittest.mock import MagicMock

    from mfg_pde.utils.validation.components import validate_components

    geom = _MockCartesianGrid(dim=1)

    # Create mock components with mismatched BC
    components = MagicMock()
    components.m_initial = np.ones(21)
    components.u_final = np.ones(21)
    components.boundary_conditions = neumann_bc(dimension=2)  # Mismatch

    result = validate_components(components, geom, require_m_initial=False, require_u_final=False)
    assert not result.is_valid
    assert any("dimension" in str(issue).lower() for issue in result.issues)
