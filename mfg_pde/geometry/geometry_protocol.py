#!/usr/bin/env python3
"""
Unified geometry protocol for MFG problems.

This module defines the protocol that all geometry objects must satisfy,
along with type enumeration and detection utilities.

Created: 2025-11-05
Part of: Unified Geometry Parameter Design
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


class GeometryType(Enum):
    """
    Enumeration of supported geometry types.

    Attributes:
        CARTESIAN_GRID: Rectangular tensor product grid
        NETWORK: Graph/network topology
        MAZE: Grid-based maze with obstacles
        DOMAIN_2D: 2D complex geometry with boundary
        DOMAIN_3D: 3D complex geometry with boundary
        IMPLICIT: Implicit geometry (level set or SDF)
        CUSTOM: User-defined custom geometry
    """

    CARTESIAN_GRID = "cartesian_grid"
    NETWORK = "network"
    MAZE = "maze"
    DOMAIN_2D = "domain_2d"
    DOMAIN_3D = "domain_3d"
    IMPLICIT = "implicit"
    CUSTOM = "custom"


@runtime_checkable
class GeometryProtocol(Protocol):
    """
    Protocol that all geometry objects must satisfy.

    This defines the minimal interface required for a geometry to be used
    with MFGProblem solvers.

    All geometry classes must implement:
        - dimension: int - Spatial dimension
        - geometry_type: GeometryType - Type of geometry
        - num_spatial_points: int - Total number of discrete spatial points
        - get_spatial_grid() - Returns grid/mesh representation
    """

    @property
    def dimension(self) -> int:
        """Spatial dimension of the geometry."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry."""
        ...

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        ...

    def get_spatial_grid(self) -> np.ndarray | list[np.ndarray]:
        """
        Get spatial grid representation.

        Returns:
            For Cartesian grids: meshgrid arrays
            For networks: adjacency matrix or node list
            For implicit: sampled points satisfying constraint
        """
        ...

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method allows each geometry type to specify how it
        should configure MFGProblem, avoiding hasattr checks and duck typing.

        Returns:
            Dictionary with keys:
                - num_spatial_points: int - Total number of spatial points
                - spatial_shape: tuple - Shape of spatial arrays
                - spatial_bounds: tuple of tuples or None - Bounds [(min, max), ...]
                - spatial_discretization: tuple or None - Discretization [Nx, Ny, ...]
                - legacy_1d_attrs: dict or None - Legacy 1D attributes (xmin, xmax, etc.)

        Added in v0.10.1 for polymorphic geometry handling.
        """
        ...


def detect_geometry_type(geometry: object) -> GeometryType:
    """
    Detect the type of a geometry object.

    Uses runtime type checking to determine what kind of geometry
    object has been provided. Checks for explicit geometry_type attribute
    first, then falls back to class name inspection.

    Args:
        geometry: Geometry object to classify

    Returns:
        GeometryType enum value

    Raises:
        ValueError: If geometry type cannot be determined

    Examples:
        >>> from mfg_pde.geometry import SimpleGrid1D
        >>> domain = SimpleGrid1D(xmin=0.0, xmax=1.0)
        >>> detect_geometry_type(domain)
        <GeometryType.CARTESIAN_GRID: 'cartesian_grid'>

        >>> from mfg_pde.geometry import NetworkGeometry
        >>> network = NetworkGeometry(topology="scale_free", n_nodes=100)
        >>> detect_geometry_type(network)
        <GeometryType.NETWORK: 'network'>
    """
    # Check for explicit geometry_type attribute
    if hasattr(geometry, "geometry_type"):
        geom_type = geometry.geometry_type
        if isinstance(geom_type, GeometryType):
            return geom_type
        elif isinstance(geom_type, str):
            return GeometryType(geom_type)

    # Fall back to class name inspection
    class_name = type(geometry).__name__.lower()

    if "network" in class_name or "graph" in class_name:
        return GeometryType.NETWORK
    elif "maze" in class_name:
        return GeometryType.MAZE
    elif "domain2d" in class_name or "domain_2d" in class_name:
        return GeometryType.DOMAIN_2D
    elif "domain3d" in class_name or "domain_3d" in class_name:
        return GeometryType.DOMAIN_3D
    elif "implicit" in class_name or "levelset" in class_name or "sdf" in class_name:
        return GeometryType.IMPLICIT
    elif any(x in class_name for x in ["domain1d", "domain_1d", "cartesian", "grid"]):
        return GeometryType.CARTESIAN_GRID
    else:
        # Default to CUSTOM for unknown types
        return GeometryType.CUSTOM


def is_geometry_compatible(geometry: object) -> bool:
    """
    Check if an object satisfies the GeometryProtocol.

    Args:
        geometry: Object to check

    Returns:
        True if object implements GeometryProtocol, False otherwise

    Examples:
        >>> from mfg_pde.geometry import SimpleGrid1D
        >>> domain = SimpleGrid1D(xmin=0.0, xmax=1.0)
        >>> is_geometry_compatible(domain)
        True

        >>> is_geometry_compatible("not a geometry")
        False
    """
    return isinstance(geometry, GeometryProtocol)


def validate_geometry(geometry: object) -> None:
    """
    Validate that an object satisfies GeometryProtocol.

    Args:
        geometry: Object to validate

    Raises:
        TypeError: If geometry does not satisfy protocol
        ValueError: If geometry has invalid properties

    Examples:
        >>> from mfg_pde.geometry import SimpleGrid1D
        >>> domain = SimpleGrid1D(xmin=0.0, xmax=1.0)
        >>> validate_geometry(domain)  # No error

        >>> validate_geometry("invalid")
        Traceback (most recent call last):
            ...
        TypeError: Object does not satisfy GeometryProtocol
    """
    if not is_geometry_compatible(geometry):
        raise TypeError(
            f"Object of type {type(geometry).__name__} does not satisfy GeometryProtocol. "
            f"Required attributes: dimension, geometry_type, num_spatial_points, get_spatial_grid()"
        )

    # Validate properties
    if not isinstance(geometry.dimension, int) or geometry.dimension < 1:
        raise ValueError(f"geometry.dimension must be a positive integer, got {geometry.dimension}")

    if not isinstance(geometry.num_spatial_points, int) or geometry.num_spatial_points < 1:
        raise ValueError(f"geometry.num_spatial_points must be a positive integer, got {geometry.num_spatial_points}")
