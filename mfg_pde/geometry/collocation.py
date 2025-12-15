"""
Geometry-aware collocation point generation for meshfree methods.

This module provides unified sampling for interior and boundary points,
with support for all geometry types (grids, meshes, implicit, CSG).

Architecture:
    Layer 2 in the collocation sampling hierarchy:
    - Uses Layer 1 (utils/numerical/particle/sampling.py) for base algorithms
    - Provides geometry-specific strategies for different domain types
    - Returns CollocationPointSet with points, boundary info, and normals

Key Classes:
    - CollocationSampler: Main interface, dispatches to geometry-specific samplers
    - CollocationPointSet: Container for points, boundary mask, normals, region IDs
    - BaseCollocationStrategy: ABC for geometry-specific implementations

Example:
    >>> from mfg_pde.geometry.collocation import CollocationSampler
    >>> from mfg_pde.geometry import Hyperrectangle
    >>>
    >>> domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
    >>> sampler = CollocationSampler(domain)
    >>> coll = sampler.generate_collocation(n_interior=400, n_boundary=100)
    >>>
    >>> # Use with GFDM solver
    >>> solver = HJBGFDMSolver(
    ...     problem,
    ...     collocation_points=coll.points,
    ...     boundary_indices=coll.boundary_indices,
    ... )

See Also:
    - Issue #482: Boundary Point Generation Module for Meshfree Methods
    - docs/development/COLLOCATION_POINT_GENERATION_REPORT.md
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mfg_pde.utils.numerical.particle.sampling import (
    MCConfig,
    PoissonDiskSampler,
    QuasiMCSampler,
    UniformMCSampler,
)

if TYPE_CHECKING:
    from mfg_pde.geometry.protocol import GeometryProtocol

logger = logging.getLogger(__name__)


@dataclass
class CollocationPointSet:
    """
    Container for collocation point data.

    This dataclass holds all information needed for meshfree methods:
    - Combined interior and boundary points
    - Boolean mask identifying boundary points
    - Outward unit normals at boundary points
    - Region IDs for multi-component boundaries (e.g., CSG domains)

    Attributes:
        points: All collocation points, shape (N, d)
        is_boundary: Boolean mask, True for boundary points, shape (N,)
        normals: Outward unit normals, zeros for interior, shape (N, d)
        region_ids: Boundary region identifier, -1 for interior, shape (N,)

    Properties:
        boundary_indices: Array of indices where is_boundary is True
        interior_indices: Array of indices where is_boundary is False
        n_interior: Number of interior points
        n_boundary: Number of boundary points
    """

    points: NDArray  # (N, d)
    is_boundary: NDArray  # (N,) bool
    normals: NDArray  # (N, d)
    region_ids: NDArray = field(default_factory=lambda: np.array([]))  # (N,) int

    def __post_init__(self):
        """Validate and set defaults."""
        if len(self.region_ids) == 0:
            # Default: all interior = -1, all boundary = 0
            self.region_ids = np.where(self.is_boundary, 0, -1)

    @property
    def boundary_indices(self) -> NDArray:
        """Indices of boundary points (for backward compatibility with GFDM)."""
        return np.where(self.is_boundary)[0]

    @property
    def interior_indices(self) -> NDArray:
        """Indices of interior points."""
        return np.where(~self.is_boundary)[0]

    @property
    def n_interior(self) -> int:
        """Number of interior points."""
        return int(np.sum(~self.is_boundary))

    @property
    def n_boundary(self) -> int:
        """Number of boundary points."""
        return int(np.sum(self.is_boundary))

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self.points.shape[1]

    def get_boundary_points(self) -> NDArray:
        """Return only boundary points."""
        return self.points[self.is_boundary]

    def get_interior_points(self) -> NDArray:
        """Return only interior points."""
        return self.points[~self.is_boundary]

    def get_boundary_normals(self) -> NDArray:
        """Return normals at boundary points only."""
        return self.normals[self.is_boundary]

    def get_region_boundary_indices(self, region_id: int) -> NDArray:
        """Get boundary indices for a specific region."""
        return np.where((self.is_boundary) & (self.region_ids == region_id))[0]


class BaseCollocationStrategy(ABC):
    """Abstract base class for geometry-specific collocation strategies."""

    def __init__(self, geometry: GeometryProtocol):
        self.geometry = geometry
        self.dimension = self._get_dimension()

    @abstractmethod
    def _get_dimension(self) -> int:
        """Get spatial dimension from geometry."""

    @abstractmethod
    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points to sample
            method: Sampling method
            seed: Random seed for reproducibility

        Returns:
            Interior points, shape (n_points, d)
        """

    @abstractmethod
    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary collocation points.

        Args:
            n_points: Number of boundary points to sample
            method: Sampling method
            seed: Random seed for reproducibility

        Returns:
            Tuple of (points, normals, region_ids):
            - points: Boundary points, shape (n_points, d)
            - normals: Outward unit normals, shape (n_points, d)
            - region_ids: Region identifiers, shape (n_points,)
        """


class CartesianGridCollocation(BaseCollocationStrategy):
    """
    Collocation strategy for Cartesian grids (TensorProductGrid).

    For Cartesian grids, points are extracted from the grid structure:
    - Interior: Grid nodes not on any boundary face
    - Boundary: Grid nodes on faces, with explicit corner handling
    """

    def _get_dimension(self) -> int:
        """Get dimension from grid."""
        if hasattr(self.geometry, "dimension"):
            return self.geometry.dimension
        elif hasattr(self.geometry, "bounds"):
            return len(self.geometry.bounds)
        raise ValueError("Cannot determine dimension from geometry")

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds."""
        if hasattr(self.geometry, "bounds"):
            return list(self.geometry.bounds)
        elif hasattr(self.geometry, "get_bounds"):
            return list(self.geometry.get_bounds())
        raise ValueError("Cannot determine bounds from geometry")

    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
    ) -> NDArray:
        """Sample interior points from grid or using specified method."""
        bounds = self._get_bounds()
        config = MCConfig(seed=seed)

        if method == "poisson_disk":
            sampler = PoissonDiskSampler(bounds, config)
        elif method == "sobol":
            sampler = QuasiMCSampler(bounds, config, "sobol")
        else:
            sampler = UniformMCSampler(bounds, config)

        return sampler.sample(n_points)

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary points on faces of Cartesian domain.

        Distributes points across all 2*d faces, with explicit corner inclusion.
        """
        bounds = self._get_bounds()
        d = self.dimension
        rng = np.random.RandomState(seed)

        # Allocate points per face (2*d faces total)
        n_faces = 2 * d
        remainder = n_points % n_faces

        all_points = []
        all_normals = []
        all_region_ids = []

        # Add corners first (always include)
        corners = self._generate_corners(bounds)
        n_corners = len(corners)
        corner_normals = self._compute_corner_normals(corners, bounds)

        all_points.append(corners)
        all_normals.append(corner_normals)
        all_region_ids.append(np.full(n_corners, -2))  # -2 for corners

        # Remaining points to distribute
        n_remaining = n_points - n_corners

        if n_remaining > 0:
            points_per_face = n_remaining // n_faces

            region_id = 0
            for dim in range(d):
                for side in [0, 1]:  # min, max
                    n_face = points_per_face + (1 if region_id < remainder else 0)
                    if n_face <= 0:
                        region_id += 1
                        continue

                    face_pts = self._sample_face(dim, side, n_face, bounds, rng)
                    normal = np.zeros(d)
                    normal[dim] = 1.0 if side == 1 else -1.0

                    all_points.append(face_pts)
                    all_normals.append(np.tile(normal, (len(face_pts), 1)))
                    all_region_ids.append(np.full(len(face_pts), region_id))

                    region_id += 1

        points = np.vstack(all_points)
        normals = np.vstack(all_normals)
        region_ids = np.concatenate(all_region_ids)

        return points, normals, region_ids

    def _generate_corners(self, bounds: list[tuple[float, float]]) -> NDArray:
        """Generate all corner vertices of the domain."""
        from itertools import product

        d = len(bounds)
        corners = []
        for indices in product([0, 1], repeat=d):
            corner = [bounds[dim][idx] for dim, idx in enumerate(indices)]
            corners.append(corner)
        return np.array(corners)

    def _compute_corner_normals(self, corners: NDArray, bounds: list[tuple[float, float]]) -> NDArray:
        """Compute averaged normals at corners."""
        d = len(bounds)
        normals = np.zeros_like(corners)

        for i, corner in enumerate(corners):
            normal = np.zeros(d)
            for dim in range(d):
                if np.isclose(corner[dim], bounds[dim][0]):
                    normal[dim] -= 1.0
                elif np.isclose(corner[dim], bounds[dim][1]):
                    normal[dim] += 1.0
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i] = normal / norm

        return normals

    def _sample_face(
        self,
        dim: int,
        side: int,
        n_points: int,
        bounds: list[tuple[float, float]],
        rng: np.random.RandomState,
    ) -> NDArray:
        """Sample points uniformly on a face."""
        d = len(bounds)
        points = np.zeros((n_points, d))

        for i in range(d):
            if i == dim:
                points[:, i] = bounds[dim][side]
            else:
                points[:, i] = rng.uniform(bounds[i][0], bounds[i][1], n_points)

        return points


class ImplicitDomainCollocation(BaseCollocationStrategy):
    """
    Collocation strategy for implicit domains defined by signed distance functions.

    Uses Newton projection to find boundary points on the zero level set.
    """

    def _get_dimension(self) -> int:
        """Get dimension from implicit domain."""
        if hasattr(self.geometry, "dimension"):
            return self.geometry.dimension
        raise ValueError("Cannot determine dimension from geometry")

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get bounding box of implicit domain."""
        if hasattr(self.geometry, "bounds"):
            bounds = self.geometry.bounds
            if isinstance(bounds, np.ndarray):
                # Shape (d, 2): each row is [min, max]
                return [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(bounds.shape[0])]
            return list(bounds)
        elif hasattr(self.geometry, "get_bounding_box"):
            bbox = self.geometry.get_bounding_box()
            # Shape (d, 2): each row is [min, max]
            return [(float(bbox[i, 0]), float(bbox[i, 1])) for i in range(bbox.shape[0])]
        raise ValueError("Cannot determine bounds from geometry")

    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample interior points using Poisson disk + rejection.

        First generates well-spaced candidates, then rejects those outside domain.
        """
        bounds = self._get_bounds()
        config = MCConfig(seed=seed)

        # Generate more candidates than needed (rejection will reduce)
        oversample_factor = 2.0
        n_candidates = int(n_points * oversample_factor)

        if method == "poisson_disk":
            sampler = PoissonDiskSampler(bounds, config)
        elif method == "sobol":
            sampler = QuasiMCSampler(bounds, config, "sobol")
        else:
            sampler = UniformMCSampler(bounds, config)

        accepted = []
        max_attempts = 10

        for _attempt in range(max_attempts):
            candidates = sampler.sample(n_candidates)

            # Check which points are inside (SDF < 0)
            sdf_values = self.geometry.signed_distance(candidates)
            inside_mask = sdf_values < 0
            inside_points = candidates[inside_mask]

            accepted.extend(inside_points.tolist())

            if len(accepted) >= n_points:
                break

            # Need more points
            n_candidates = int((n_points - len(accepted)) * oversample_factor)

        result = np.array(accepted[:n_points])
        logger.debug(f"ImplicitCollocation: generated {len(result)} interior points")
        return result

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary points using Newton projection to zero level set.

        Algorithm:
        1. Generate interior seed points
        2. Project each to boundary via Newton iteration on SDF
        3. Compute normals from SDF gradient
        """
        # Generate seed points inside domain
        seeds = self.sample_interior(n_points, method="sobol", seed=seed)

        # Project to boundary
        boundary_points = self._project_to_boundary(seeds)

        # Compute normals from SDF gradient
        normals = self._compute_normals(boundary_points)

        # All same region for single implicit domain
        region_ids = np.zeros(len(boundary_points), dtype=int)

        return boundary_points, normals, region_ids

    def _project_to_boundary(
        self,
        points: NDArray,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> NDArray:
        """Project points to boundary using Newton iteration."""
        projected = points.copy()

        for _ in range(max_iter):
            # Evaluate SDF and gradient
            sdf = self.geometry.signed_distance(projected)
            grad = self._compute_sdf_gradient(projected)

            # Newton step: x_new = x - φ(x) * ∇φ(x) / |∇φ(x)|²
            grad_norm_sq = np.sum(grad**2, axis=1, keepdims=True)
            grad_norm_sq = np.maximum(grad_norm_sq, 1e-12)  # Avoid division by zero

            step = (sdf[:, np.newaxis] * grad) / grad_norm_sq
            projected = projected - step

            # Check convergence
            if np.max(np.abs(sdf)) < tol:
                break

        return projected

    def _compute_sdf_gradient(self, points: NDArray, eps: float = 1e-6) -> NDArray:
        """Compute SDF gradient using finite differences (fallback)."""
        # Try analytical gradient first
        if hasattr(self.geometry, "signed_distance_gradient"):
            return self.geometry.signed_distance_gradient(points)

        # Finite difference fallback
        d = points.shape[1]
        grad = np.zeros_like(points)

        for dim in range(d):
            points_plus = points.copy()
            points_minus = points.copy()
            points_plus[:, dim] += eps
            points_minus[:, dim] -= eps

            sdf_plus = self.geometry.signed_distance(points_plus)
            sdf_minus = self.geometry.signed_distance(points_minus)

            grad[:, dim] = (sdf_plus - sdf_minus) / (2 * eps)

        return grad

    def _compute_normals(self, points: NDArray) -> NDArray:
        """Compute outward unit normals at boundary points."""
        grad = self._compute_sdf_gradient(points)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return grad / norms


class CollocationSampler:
    """
    Unified collocation sampler for any geometry type.

    Automatically selects the appropriate strategy based on geometry type:
    - TensorProductGrid → CartesianGridCollocation
    - ImplicitDomain/Hypersphere/Hyperrectangle → ImplicitDomainCollocation
    - CSG domains → CSGDomainCollocation (delegates to components)

    Example:
        >>> from mfg_pde.geometry import Hyperrectangle
        >>> domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
        >>> sampler = CollocationSampler(domain)
        >>> coll = sampler.generate_collocation(n_interior=400, n_boundary=100)
        >>> print(f"Generated {coll.n_interior} interior, {coll.n_boundary} boundary")
    """

    def __init__(self, geometry: GeometryProtocol):
        """
        Initialize collocation sampler.

        Args:
            geometry: Geometry object supporting the GeometryProtocol
        """
        self.geometry = geometry
        self._strategy = self._select_strategy()

    def _select_strategy(self) -> BaseCollocationStrategy:
        """Select appropriate strategy based on geometry type."""
        geom = self.geometry

        # Check for TensorProductGrid
        if hasattr(geom, "get_grid_points") or type(geom).__name__ == "TensorProductGrid":
            return CartesianGridCollocation(geom)

        # Check for implicit domain (has signed_distance method)
        if hasattr(geom, "signed_distance"):
            # Check for CSG composite
            if hasattr(geom, "domains") or hasattr(geom, "base_domain"):
                # TODO: Implement CSGDomainCollocation
                return ImplicitDomainCollocation(geom)
            return ImplicitDomainCollocation(geom)

        # Check for mesh (has vertices)
        if hasattr(geom, "vertices"):
            # TODO: Implement MeshCollocation
            raise NotImplementedError("MeshCollocation not yet implemented")

        # Fallback: try to treat as bounding box
        if hasattr(geom, "bounds") or hasattr(geom, "get_bounds"):
            return CartesianGridCollocation(geom)

        raise ValueError(f"Unsupported geometry type: {type(geom)}")

    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points
            method: Sampling method ("poisson_disk", "sobol", "uniform")
            seed: Random seed

        Returns:
            Interior points, shape (n_points, d)
        """
        return self._strategy.sample_interior(n_points, method, seed)

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Sample boundary collocation points.

        Args:
            n_points: Number of boundary points
            method: Sampling method
            seed: Random seed

        Returns:
            Tuple of (points, normals)
        """
        points, normals, _ = self._strategy.sample_boundary(n_points, method, seed)
        return points, normals

    def generate_collocation(
        self,
        n_interior: int,
        n_boundary: int,
        interior_method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        boundary_method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> CollocationPointSet:
        """
        Generate complete collocation point set.

        Args:
            n_interior: Number of interior points
            n_boundary: Number of boundary points
            interior_method: Sampling method for interior
            boundary_method: Sampling method for boundary
            seed: Random seed

        Returns:
            CollocationPointSet with combined points and metadata
        """
        # Sample interior
        interior = self.sample_interior(n_interior, interior_method, seed)

        # Sample boundary
        boundary, normals_bdy, region_ids_bdy = self._strategy.sample_boundary(n_boundary, boundary_method, seed)

        # Combine
        n_int = len(interior)
        n_bdy = len(boundary)
        d = interior.shape[1]

        points = np.vstack([interior, boundary])
        is_boundary = np.concatenate([np.zeros(n_int, dtype=bool), np.ones(n_bdy, dtype=bool)])
        normals = np.vstack([np.zeros((n_int, d)), normals_bdy])
        region_ids = np.concatenate([np.full(n_int, -1), region_ids_bdy])

        return CollocationPointSet(
            points=points,
            is_boundary=is_boundary,
            normals=normals,
            region_ids=region_ids,
        )


# Convenience function
def generate_collocation(
    geometry: GeometryProtocol,
    n_interior: int,
    n_boundary: int,
    **kwargs,
) -> CollocationPointSet:
    """
    Convenience function to generate collocation points.

    Args:
        geometry: Geometry object
        n_interior: Number of interior points
        n_boundary: Number of boundary points
        **kwargs: Additional arguments passed to CollocationSampler.generate_collocation

    Returns:
        CollocationPointSet
    """
    sampler = CollocationSampler(geometry)
    return sampler.generate_collocation(n_interior, n_boundary, **kwargs)
