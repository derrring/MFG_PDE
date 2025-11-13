"""
Point cloud geometry for particle-based MFG solvers.

This module provides a lightweight geometry wrapper for particle arrays,
enabling particle-particle projection in hybrid MFG solvers (e.g., GFDM + Particle FP).

Two patterns are supported:
1. Fixed collocation + moving particles: Requires projection between different point sets
2. Co-moving particles: Identity projection when same particle set is used

Created: 2025-11-12
Part of: Issue #269 - Particle-Particle Projection Support
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.geometry.protocol import GeometryProtocol, GeometryType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PointCloudGeometry:
    """
    Lightweight geometry wrapper for particle arrays.

    This class wraps particle position arrays to enable them to be used with
    the GeometryProjector system for particle-particle projections.

    Attributes:
        positions: Particle positions (N, dimension)
        dimension: Spatial dimension
        num_particles: Number of particles
        metadata: Optional metadata (e.g., particle IDs, weights)

    Examples:
        >>> # Fixed collocation points for GFDM
        >>> collocation = np.random.uniform(0, 1, (500, 2))
        >>> hjb_geometry = PointCloudGeometry(collocation)
        >>>
        >>> # Evolving particles for FP
        >>> particles = np.random.uniform(0, 1, (1000, 2))
        >>> fp_geometry = PointCloudGeometry(particles)
        >>>
        >>> # Use with GeometryProjector
        >>> from mfg_pde.geometry import GeometryProjector
        >>> projector = GeometryProjector(hjb_geometry, fp_geometry)
        >>> U_on_particles = projector.project_hjb_to_fp(U_on_collocation)
    """

    def __init__(
        self,
        positions: NDArray[np.floating],
        metadata: dict | None = None,
    ):
        """
        Initialize point cloud geometry.

        Args:
            positions: Particle positions (N, dimension) or (N,) for 1D
            metadata: Optional metadata dict (particle IDs, weights, etc.)

        Raises:
            ValueError: If positions have invalid shape
        """
        self.positions = np.asarray(positions, dtype=float)

        # Handle 1D case
        if self.positions.ndim == 1:
            self.positions = self.positions.reshape(-1, 1)

        if self.positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array (N, dimension), got shape {self.positions.shape}")

        self.metadata = metadata or {}

    @property
    def dimension(self) -> int:
        """Spatial dimension of the point cloud."""
        return self.positions.shape[1]

    @property
    def num_particles(self) -> int:
        """Number of particles in the cloud."""
        return self.positions.shape[0]

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (alias for num_particles)."""
        return self.num_particles

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (CUSTOM for point clouds)."""
        return GeometryType.CUSTOM

    def get_spatial_grid(self) -> NDArray[np.floating]:
        """
        Get spatial grid representation (returns particle positions).

        Returns:
            Particle positions (N, dimension)
        """
        return self.positions

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of particles
                - spatial_shape: (num_particles,)
                - spatial_bounds: Bounding box of particles
                - spatial_discretization: None (not applicable)
                - legacy_1d_attrs: None
        """
        min_coords = np.min(self.positions, axis=0)
        max_coords = np.max(self.positions, axis=0)
        bounds = tuple((float(mn), float(mx)) for mn, mx in zip(min_coords, max_coords, strict=True))

        return {
            "num_spatial_points": self.num_particles,
            "spatial_shape": (self.num_particles,),
            "spatial_bounds": bounds,
            "spatial_discretization": None,
            "legacy_1d_attrs": None,
        }

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Bounding box of the point cloud: (min_coords, max_coords)."""
        return np.min(self.positions, axis=0), np.max(self.positions, axis=0)

    def is_same_pointset(self, other: PointCloudGeometry, tol: float = 1e-10) -> bool:
        """
        Check if two point clouds represent the same particle set.

        This enables identity projection optimization when HJB and FP use the
        same particles (co-moving particle pattern).

        Args:
            other: Another PointCloudGeometry
            tol: Tolerance for position comparison

        Returns:
            True if point clouds have same positions (up to tolerance)

        Examples:
            >>> # Co-moving particles (identity projection)
            >>> particles = np.random.uniform(0, 1, (1000, 2))
            >>> geom1 = PointCloudGeometry(particles)
            >>> geom2 = PointCloudGeometry(particles)  # Same array
            >>> geom1.is_same_pointset(geom2)
            True
            >>>
            >>> # Different particles (need projection)
            >>> collocation = np.random.uniform(0, 1, (500, 2))
            >>> geom3 = PointCloudGeometry(collocation)
            >>> geom1.is_same_pointset(geom3)
            False
        """
        if not isinstance(other, PointCloudGeometry):
            return False

        if self.num_particles != other.num_particles:
            return False

        if self.dimension != other.dimension:
            return False

        # Check if arrays are identical (same object)
        if self.positions is other.positions:
            return True

        # Check if positions are numerically equal
        return np.allclose(self.positions, other.positions, atol=tol, rtol=0.0)

    def __repr__(self) -> str:
        """String representation."""
        return f"PointCloudGeometry(num_particles={self.num_particles}, dimension={self.dimension})"


# Register with GeometryProtocol (runtime checkable)
assert isinstance(PointCloudGeometry(np.random.rand(10, 2)), GeometryProtocol), (
    "PointCloudGeometry must implement GeometryProtocol"
)


__all__ = ["PointCloudGeometry"]
