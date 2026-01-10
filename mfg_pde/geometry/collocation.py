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

# Issue #543: Runtime import for isinstance() checks
from mfg_pde.geometry.protocol import GeometryProtocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mfg_pde.utils.numerical.particle.sampling import (
    MCConfig,
    PoissonDiskSampler,
    QuasiMCSampler,
    UniformMCSampler,
)

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
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points to sample
            method: Sampling method:
                - "lloyd" (recommended): CVT/Lloyd relaxation for quasi-uniform
                - "poisson_disk": Blue noise (poor for non-convex domains)
                - "sobol": Quasi-random sequence
                - "uniform": Pure random
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
        # Issue #543: Use GeometryProtocol instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(
                f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}. "
                f"All geometries must provide dimension property."
            )
        return self.geometry.dimension

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds."""
        # Issue #543: Use GeometryProtocol.get_bounds() instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")

        bounds_tuple = self.geometry.get_bounds()
        if bounds_tuple is None:
            raise ValueError("Geometry does not have bounded domain (get_bounds() returned None)")

        # Convert from (min_coords, max_coords) to list of (min, max) tuples
        min_coords, max_coords = bounds_tuple
        return [(float(min_coords[i]), float(max_coords[i])) for i in range(len(min_coords))]

    def sample_interior(
        self,
        n_points: int,
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """Sample interior points using specified method.

        Args:
            n_points: Number of points to generate
            method: Sampling method:
                - "lloyd": CVT/Lloyd relaxation (best quasi-uniform)
                - "poisson_disk": Blue noise with minimum spacing
                - "sobol": Quasi-random low-discrepancy
                - "uniform": Pure random
            seed: Random seed
            refine_steps: Lloyd iterations (only for method="lloyd")
        """
        bounds = self._get_bounds()
        config = MCConfig(seed=seed)

        if method == "lloyd":
            # Initialize with Sobol, then apply Lloyd relaxation
            sampler = QuasiMCSampler(bounds, config, "sobol")
            points = sampler.sample(n_points)
            if refine_steps > 0 and len(points) > 1:
                points = self._refine_lloyd_cartesian(points, bounds, refine_steps)
            return points
        elif method == "poisson_disk":
            sampler = PoissonDiskSampler(bounds, config)
        elif method == "sobol":
            sampler = QuasiMCSampler(bounds, config, "sobol")
        else:
            sampler = UniformMCSampler(bounds, config)

        return sampler.sample(n_points)

    def _refine_lloyd_cartesian(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 20,
    ) -> NDArray:
        """Lloyd/CVT relaxation for Cartesian domains (no SDF needed)."""
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)
        interaction_radius = 2.5 * target_spacing

        for step in range(steps):
            forces = np.zeros_like(current)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)
                force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # Boundary repulsion for Cartesian box
            for dim in range(d):
                lo, hi = bounds[dim]
                margin = target_spacing * 1.5
                # Near lower bound
                near_lo = current[:, dim] < lo + margin
                for i in np.where(near_lo)[0]:
                    dist_to_wall = max(current[i, dim] - lo, 1e-10)
                    forces[i, dim] += 0.5 / (dist_to_wall**2)
                # Near upper bound
                near_hi = current[:, dim] > hi - margin
                for i in np.where(near_hi)[0]:
                    dist_to_wall = max(hi - current[i, dim], 1e-10)
                    forces[i, dim] -= 0.5 / (dist_to_wall**2)

            # Adaptive time step with annealing
            max_force = np.max(np.linalg.norm(forces, axis=1))
            anneal = 1.0 - 0.5 * (step / steps)
            dt = min(0.15 * anneal, 0.4 * target_spacing / (max_force + 1e-10))

            current = current + forces * dt

            # Clamp to bounds
            for dim in range(d):
                current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + 1e-6, bounds[dim][1] - 1e-6)

        return current

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

    Sampling Methods:
        - "lloyd" (recommended): CVT/Lloyd relaxation via particle repulsion.
          Best for non-convex domains. Achieves quasi-uniform coverage.
        - "sobol": Quasi-random low-discrepancy sequence + rejection sampling.
          Good for convex domains, may have gaps in non-convex regions.
        - "poisson_disk": Blue noise with minimum spacing guarantee.
          WARNING: Poor coverage in non-convex geometries (thin corridors,
          regions separated by obstacles). Use "lloyd" instead.
        - "uniform": Pure random sampling. Not recommended.

    For non-convex domains (CSG with obstacles, L-shaped regions, etc.),
    always use method="lloyd" which applies particle repulsion to fill
    all regions uniformly regardless of geometry complexity.
    """

    def _get_dimension(self) -> int:
        """Get dimension from implicit domain."""
        # Issue #543: Use GeometryProtocol instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")
        return self.geometry.dimension

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get bounding box of implicit domain."""
        # Issue #543: Use GeometryProtocol.get_bounds() instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")

        bounds_tuple = self.geometry.get_bounds()
        if bounds_tuple is None:
            raise ValueError("Implicit domain must have bounded domain for collocation")

        # Convert from (min_coords, max_coords) to list of (min, max) tuples
        min_coords, max_coords = bounds_tuple
        return [(float(min_coords[i]), float(max_coords[i])) for i in range(len(min_coords))]

    def sample_interior(
        self,
        n_points: int,
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """
        Sample interior points for implicit domain.

        Args:
            n_points: Number of points to generate
            method: Sampling method:
                - "lloyd" (default, recommended): CVT/Lloyd relaxation via particle
                  repulsion. Best quasi-uniform coverage for any geometry.
                - "poisson_disk": Blue noise. WARNING: poor for non-convex domains.
                - "sobol": Quasi-random sequence + rejection.
                - "uniform": Pure random (not recommended).
            seed: Random seed for reproducibility
            refine_steps: Number of Lloyd/particle repulsion iterations.
                For method="lloyd", this is the main algorithm (default 20).
                For other methods, this refines the initial distribution.
                Set to 0 to disable refinement (not recommended for non-convex).

        Returns:
            Interior points, shape (n_points, d)

        Note:
            For non-convex domains (obstacles, thin corridors, L-shapes),
            use method="lloyd" which fills all regions uniformly via
            particle repulsion regardless of geometry complexity.
        """
        bounds = self._get_bounds()
        rng = np.random.RandomState(seed)

        if method == "lloyd":
            # Lloyd/CVT: Initialize with rejection sampling, then heavy refinement
            points = self._sample_initial_rejection(n_points, bounds, rng)
            # Lloyd uses more iterations for better convergence
            lloyd_steps = max(refine_steps, 30)
            if len(points) > 1:
                points = self._refine_lloyd_cvt(points, bounds, lloyd_steps)
        else:
            # Other methods: initial sampling + optional refinement
            if method == "poisson_disk":
                logger.warning(
                    "Poisson disk may have poor coverage in non-convex domains. "
                    "Consider method='lloyd' for better uniformity."
                )
            points = self._sample_initial_rejection(n_points, bounds, rng)

            # Refine via particle repulsion
            if refine_steps > 0 and len(points) > 1:
                points = self._refine_particle_repulsion(points, bounds, refine_steps)

        logger.debug(f"ImplicitCollocation: {len(points)} points, method={method}")
        return points

    def _sample_initial_rejection(
        self,
        n_points: int,
        bounds: list[tuple[float, float]],
        rng: np.random.RandomState,
    ) -> NDArray:
        """Generate initial points via simple rejection sampling."""
        accepted = []
        max_attempts = n_points * 20

        for _ in range(max_attempts):
            if len(accepted) >= n_points:
                break
            candidate = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            if self.geometry.signed_distance(candidate.reshape(1, -1))[0] < 0:
                accepted.append(candidate)

        return np.array(accepted[:n_points])

    def _refine_particle_repulsion(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 20,
        dt: float = 0.1,
    ) -> NDArray:
        """
        Refine point distribution using particle repulsion (energy minimization).

        Minimizes the energy functional:
            E = sum_{i!=j} 1/||x_i - x_j||^p + sum_i 1/d(x_i, boundary)^q

        Points naturally "flow" into non-convex regions, solving the thin-wall
        problem that plagues Poisson disk sampling.

        Reference: Lloyd's relaxation / Centroidal Voronoi Tessellation
        """
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        # Estimate interaction radius from point density
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)
        interaction_radius = 3.0 * target_spacing

        for _step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion (via KDTree for efficiency)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsion: 1/r^2 kernel (adjustable)
                force_mag = 1.0 / (dist**2 + 1e-6)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Boundary repulsion (push away from walls)
            sdf = self.geometry.signed_distance(current)
            grad_sdf = self._compute_sdf_gradient(current)

            # Repel from boundary: force proportional to 1/d^2
            # Only apply when close to boundary (within target_spacing)
            close_to_boundary = np.abs(sdf) < target_spacing
            for i in np.where(close_to_boundary)[0]:
                boundary_dist = np.abs(sdf[i])
                if boundary_dist < 1e-10:
                    boundary_dist = 1e-10
                # Push inward (opposite to gradient which points outward)
                force_mag = 0.5 / (boundary_dist**2)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])

            # 3. Adaptive time step (reduce if forces are large)
            max_force = np.max(np.linalg.norm(forces, axis=1))
            effective_dt = min(dt, 0.5 * target_spacing / (max_force + 1e-10))

            # 4. Update positions
            current = current + forces * effective_dt

            # 5. Project back to domain interior if pushed outside
            sdf_new = self.geometry.signed_distance(current)
            outside = sdf_new >= 0

            if np.any(outside):
                # Newton projection back to boundary, then slight inward push
                grad = self._compute_sdf_gradient(current[outside])
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                # Project to boundary then push slightly inside
                current[outside] -= (sdf_new[outside, np.newaxis] + 0.01 * target_spacing) * grad / grad_norm

            # 6. Clamp to bounds
            for i in range(d):
                current[:, i] = np.clip(current[:, i], bounds[i][0] + 1e-6, bounds[i][1] - 1e-6)

        return current

    def _refine_lloyd_cvt(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 30,
    ) -> NDArray:
        """
        Lloyd/CVT relaxation for quasi-uniform point distribution.

        This is the recommended method for non-convex domains. Uses particle
        repulsion with adaptive parameters optimized for CVT convergence.

        Algorithm:
            1. Inter-particle repulsion (inverse-square kernel)
            2. Boundary repulsion (keeps points inside domain)
            3. Adaptive time stepping for stability
            4. Project outside points back to domain

        The result approximates a Centroidal Voronoi Tessellation (CVT) where
        each point is at the centroid of its Voronoi cell, achieving optimal
        quasi-uniform coverage.

        Args:
            points: Initial point positions, shape (N, d)
            bounds: Bounding box [(min, max), ...] per dimension
            steps: Number of Lloyd iterations (30 recommended for convergence)

        Returns:
            Refined points with quasi-uniform spacing, shape (N, d)

        Reference:
            Lloyd, S. (1982). "Least squares quantization in PCM."
            Du, Faber, Gunzburger (1999). "Centroidal Voronoi Tessellations."
        """
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        # Compute target spacing from point density
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)

        # Interaction radius: affects how far particles "see" each other
        # Larger = smoother but slower; smaller = faster but may have gaps
        interaction_radius = 2.5 * target_spacing

        # Adaptive parameters for stable convergence
        dt_base = 0.15  # Base time step
        boundary_strength = 0.8  # Boundary repulsion strength

        for step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion via KDTree
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Identical points: apply random perturbation
                    diff = np.random.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)

                # Inverse-square repulsion (softer than 1/r^3)
                force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Boundary repulsion using SDF
            sdf = self.geometry.signed_distance(current)
            grad_sdf = self._compute_sdf_gradient(current)

            # Apply boundary repulsion when close to boundary
            boundary_zone = target_spacing * 1.5
            close_mask = np.abs(sdf) < boundary_zone

            for i in np.where(close_mask)[0]:
                boundary_dist = max(np.abs(sdf[i]), 1e-10)
                # Repulsion strength increases near boundary
                force_mag = boundary_strength / (boundary_dist**2)
                # Push inward (opposite to SDF gradient)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])

            # 3. Adaptive time step (decreases as we converge)
            max_force = np.max(np.linalg.norm(forces, axis=1))
            # Anneal dt: larger early, smaller late for fine-tuning
            anneal_factor = 1.0 - 0.5 * (step / steps)
            dt = min(dt_base * anneal_factor, 0.4 * target_spacing / (max_force + 1e-10))

            # 4. Update positions
            current = current + forces * dt

            # 5. Project outside points back to domain interior
            sdf_new = self.geometry.signed_distance(current)
            outside = sdf_new >= 0

            if np.any(outside):
                grad = self._compute_sdf_gradient(current[outside])
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                # Project to boundary, then push slightly inside
                push_dist = 0.02 * target_spacing
                current[outside] -= (sdf_new[outside, np.newaxis] + push_dist) * grad / grad_norm

            # 6. Clamp to bounding box
            for dim in range(d):
                margin = 1e-6
                current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + margin, bounds[dim][1] - margin)

        return current

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
        refine_steps: int = 30,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary points with geodesic repulsion refinement.

        Algorithm:
        1. Generate interior seed points
        2. Project each to boundary via Newton iteration on SDF
        3. Refine: Apply tangential repulsion for uniform spacing along boundary
        4. Compute normals from SDF gradient

        The refinement step moves points along the boundary (tangent direction)
        to achieve uniform spacing, solving clustering from projection.

        Args:
            n_points: Number of boundary points
            method: Initial seed generation method
            seed: Random seed
            refine_steps: Boundary repulsion iterations (0 to disable)
        """
        # Generate seed points inside domain
        seeds = self.sample_interior(n_points, method="sobol", seed=seed, refine_steps=0)

        # Project to boundary
        boundary_points = self._project_to_boundary(seeds)

        # Refine via tangential repulsion (geodesic relaxation)
        if refine_steps > 0 and len(boundary_points) > 1:
            boundary_points = self._refine_boundary_repulsion(boundary_points, refine_steps)

        # Compute normals from SDF gradient
        normals = self._compute_normals(boundary_points)

        # All same region for single implicit domain
        region_ids = np.zeros(len(boundary_points), dtype=int)

        return boundary_points, normals, region_ids

    def _refine_boundary_repulsion(
        self,
        points: NDArray,
        steps: int = 30,
        dt: float = 0.05,
    ) -> NDArray:
        """
        Refine boundary point distribution using tangential repulsion.

        Points are constrained to the boundary manifold (SDF = 0).
        Forces are projected to the tangent space before update.
        After each step, points are re-projected to ensure SDF = 0.

        This achieves uniform spacing along the boundary curve/surface.
        """
        from scipy.spatial import cKDTree

        N, _ = points.shape
        current = points.copy()

        # Estimate target spacing from arc length / perimeter approximation
        # Use mean distance between consecutive sorted points as estimate
        target_spacing = self._estimate_boundary_spacing(current, N)
        interaction_radius = 4.0 * target_spacing

        for _step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion (KDTree)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsion: 1/r^2 kernel
                force_mag = 1.0 / (dist**2 + 1e-6)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Project forces to tangent space (remove normal component)
            normals = self._compute_normals(current)
            for i in range(N):
                normal_component = np.dot(forces[i], normals[i]) * normals[i]
                forces[i] = forces[i] - normal_component  # Tangential only

            # 3. Adaptive time step
            max_force = np.max(np.linalg.norm(forces, axis=1))
            effective_dt = min(dt, 0.3 * target_spacing / (max_force + 1e-10))

            # 4. Update positions (tangential movement)
            current = current + forces * effective_dt

            # 5. Re-project to boundary (maintain SDF = 0)
            current = self._project_to_boundary(current, max_iter=5, tol=1e-7)

        return current

    def _estimate_boundary_spacing(self, points: NDArray, n_points: int) -> float:
        """Estimate target spacing for boundary points."""
        bounds = self._get_bounds()
        d = len(bounds)

        if d == 2:
            # Rough perimeter estimate: sum of bounding box edges
            perimeter = 2 * sum(b[1] - b[0] for b in bounds)
            return perimeter / n_points
        else:
            # Higher dimensions: use bounding box surface area estimate
            bbox_dims = np.array([b[1] - b[0] for b in bounds])
            surface_area = 2 * sum(np.prod(np.delete(bbox_dims, i)) for i in range(d))
            return (surface_area / n_points) ** (1.0 / (d - 1))

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
        # Issue #543: Use try/except for optional method instead of hasattr()
        # Try analytical gradient first
        try:
            return self.geometry.signed_distance_gradient(points)
        except AttributeError:
            pass

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
        # Issue #543: Use try/except and concrete types instead of hasattr()
        geom = self.geometry

        # Check for TensorProductGrid by name (concrete type check)
        if type(geom).__name__ == "TensorProductGrid":
            return CartesianGridCollocation(geom)

        # Check for grid-like geometry (has get_grid_points method)
        try:
            _ = geom.get_grid_points
            return CartesianGridCollocation(geom)
        except AttributeError:
            pass

        # Check for implicit domain (has signed_distance method)
        try:
            _ = geom.signed_distance
            # Check for CSG composite (has domains or base_domain)
            try:
                _ = geom.domains
                # TODO: Implement CSGDomainCollocation
                return ImplicitDomainCollocation(geom)
            except AttributeError:
                try:
                    _ = geom.base_domain
                    # TODO: Implement CSGDomainCollocation
                    return ImplicitDomainCollocation(geom)
                except AttributeError:
                    pass
            return ImplicitDomainCollocation(geom)
        except AttributeError:
            pass

        # Check for mesh (has vertices)
        try:
            _ = geom.vertices
            # TODO: Implement MeshCollocation
            raise NotImplementedError("MeshCollocation not yet implemented")
        except AttributeError:
            pass

        # Fallback: use GeometryProtocol bounds
        if isinstance(geom, GeometryProtocol):
            bounds = geom.get_bounds()
            if bounds is not None:
                return CartesianGridCollocation(geom)

        raise ValueError(
            f"Unsupported geometry type: {type(geom).__name__}. "
            f"Geometry must implement GeometryProtocol with valid bounds, "
            f"or provide signed_distance, get_grid_points, or vertices methods."
        )

    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points
            method: Sampling method ("poisson_disk", "sobol", "uniform")
            seed: Random seed
            refine_steps: Particle repulsion iterations (0 to disable)

        Returns:
            Interior points, shape (n_points, d)
        """
        # Check if strategy supports refine_steps
        import inspect

        sig = inspect.signature(self._strategy.sample_interior)
        if "refine_steps" in sig.parameters:
            return self._strategy.sample_interior(n_points, method, seed, refine_steps)
        return self._strategy.sample_interior(n_points, method, seed)

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
        refine_steps: int = 30,
    ) -> tuple[NDArray, NDArray]:
        """
        Sample boundary collocation points.

        Args:
            n_points: Number of boundary points
            method: Sampling method
            seed: Random seed
            refine_steps: Boundary repulsion iterations (0 to disable)

        Returns:
            Tuple of (points, normals)
        """
        # Check if strategy supports refine_steps
        import inspect

        sig = inspect.signature(self._strategy.sample_boundary)
        if "refine_steps" in sig.parameters:
            points, normals, _ = self._strategy.sample_boundary(n_points, method, seed, refine_steps)
        else:
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
