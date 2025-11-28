"""
Meshfree boundary condition applicator for particle and collocation methods.

This module provides BC application for meshfree discretizations:
- Particle methods (Lagrangian particle FP solver)
- Collocation methods (RBF, GFDM)
- High-dimensional implicit domains (d >= 4)

BC Types Supported:
- Reflecting: Particles bounce off boundary (project back to interior)
- Absorbing: Particles at boundary are removed
- Periodic: Particles wrap around (hyperrectangle domains only)
- Penalty: Add penalty term to field values at boundary (collocation)
- Projection: Project field values at boundary nodes

Architecture:
    Domain provides geometry via GeometryProtocol (boundary detection, projection)
    MeshfreeApplicator applies BC based on domain's boundary info

Created: 2025-11-27
Part of: Unified Boundary Condition Architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from mfg_pde.geometry.protocol import GeometryProtocol

# Import base class for inheritance
from .applicator_base import BaseMeshfreeApplicator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MeshfreeApplicator(BaseMeshfreeApplicator):
    """
    Boundary condition applicator for meshfree methods.

    Works with any geometry implementing GeometryProtocol, using the geometry's
    boundary methods to detect and handle boundary conditions.

    Supports two modes:
    1. Particle BC: Apply BC to particle positions (reflecting, absorbing, periodic)
    2. Field BC: Apply BC to field values at collocation points (penalty, projection)

    Examples:
        >>> from mfg_pde.geometry import Hyperrectangle
        >>> domain = Hyperrectangle([[0, 1], [0, 1]])
        >>> applicator = MeshfreeApplicator(domain)
        >>>
        >>> # Particle BC
        >>> particles = np.array([[0.5, 0.5], [1.2, 0.3]])  # One outside
        >>> particles_updated = applicator.apply_particle_bc(particles, "reflecting")
        >>> assert np.all(domain.contains(particles_updated))
        >>>
        >>> # Field BC at collocation points
        >>> u = np.random.rand(100)  # Field at 100 collocation points
        >>> points = domain.sample_interior(100)
        >>> u_with_bc = applicator.apply_field_bc(u, points, bc_type="dirichlet", bc_value=0.0)
    """

    def __init__(self, geometry: GeometryProtocol):
        """
        Initialize meshfree BC applicator.

        Args:
            geometry: Geometry implementing GeometryProtocol with boundary methods
        """
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(geometry).__name__}")

        super().__init__(dimension=geometry.dimension)
        self.geometry = geometry

    # =========================================================================
    # Particle Boundary Conditions
    # =========================================================================

    def apply_particle_bc(
        self,
        particles: NDArray[np.floating],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to particles.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Project particles back into domain
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles (hyperrectangle only)

        Returns:
            Updated particle positions (may have fewer particles for "absorbing")

        Examples:
            >>> particles = np.array([[0.5, 0.5], [1.2, 0.3]])
            >>> updated = applicator.apply_particle_bc(particles, "reflecting")
        """
        particles = np.atleast_2d(particles)

        if bc_type == "reflecting":
            return self._apply_reflecting_bc(particles)

        elif bc_type == "absorbing":
            return self._apply_absorbing_bc(particles)

        elif bc_type == "periodic":
            return self._apply_periodic_bc(particles)

        else:
            raise ValueError(f"Unknown particle BC type: {bc_type}")

    def _apply_reflecting_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project particles outside domain back to interior."""
        return self.geometry.project_to_interior(particles)

    def _apply_absorbing_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Remove particles outside domain."""
        on_boundary = self.geometry.is_on_boundary(particles)

        bounds = self.geometry.get_bounds()
        if bounds is None:
            # Unbounded domain - keep all particles
            return particles

        min_coords, max_coords = bounds
        inside = np.ones(len(particles), dtype=bool)

        for d in range(self.dimension):
            inside &= (particles[:, d] >= min_coords[d]) & (particles[:, d] <= max_coords[d])

        # For absorbing BC, remove particles at boundary too
        inside &= ~on_boundary

        return particles[inside]

    def _apply_periodic_bc(
        self,
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Wrap particles around domain (periodic BC)."""
        bounds = self.geometry.get_bounds()
        if bounds is None:
            raise ValueError("Periodic BC requires bounded domain")

        min_coords, max_coords = bounds
        domain_size = max_coords - min_coords

        # Wrap particles using modulo
        result = particles.copy()
        for d in range(self.dimension):
            result[:, d] = ((particles[:, d] - min_coords[d]) % domain_size[d]) + min_coords[d]

        return result

    # =========================================================================
    # Field Boundary Conditions (for collocation methods)
    # =========================================================================

    def apply_field_bc(
        self,
        field: NDArray[np.floating],
        points: NDArray[np.floating],
        bc_type: Literal["dirichlet", "neumann", "penalty"] = "dirichlet",
        bc_value: float | Callable[[NDArray], float] = 0.0,
        penalty_weight: float = 1e6,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to field values at collocation points.

        Args:
            field: Field values at collocation points - shape (N,)
            points: Collocation point positions - shape (N, d)
            bc_type: Boundary condition type
                - "dirichlet": Set field to bc_value at boundary
                - "neumann": Set normal derivative to bc_value (requires additional ops)
                - "penalty": Add penalty term pushing field toward bc_value
            bc_value: Boundary value (float or callable(points) -> float)
            penalty_weight: Weight for penalty method

        Returns:
            Updated field values with BC applied

        Examples:
            >>> u = np.random.rand(100)
            >>> points = domain.sample_interior(100)
            >>> u_bc = applicator.apply_field_bc(u, points, "dirichlet", bc_value=0.0)
        """
        field = field.copy()
        points = np.atleast_2d(points)

        # Find boundary points
        on_boundary = self.geometry.is_on_boundary(points)

        if not np.any(on_boundary):
            return field

        # Get boundary value
        if callable(bc_value):
            boundary_points = points[on_boundary]
            boundary_values = np.array([bc_value(p) for p in boundary_points])
        else:
            boundary_values = bc_value

        if bc_type == "dirichlet":
            field[on_boundary] = boundary_values

        elif bc_type == "penalty":
            # Soft enforcement via penalty
            field[on_boundary] = (field[on_boundary] + penalty_weight * boundary_values) / (1 + penalty_weight)

        elif bc_type == "neumann":
            # Neumann BC requires derivative info - not directly applicable to field values
            # This would need integration with the solver's derivative operators
            raise NotImplementedError(
                "Neumann BC for collocation requires solver-specific implementation. "
                "Use the solver's derivative operators to enforce Neumann conditions."
            )

        else:
            raise ValueError(f"Unknown field BC type: {bc_type}")

        return field

    # =========================================================================
    # Boundary Information Queries
    # =========================================================================

    def get_boundary_mask(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.bool_]:
        """
        Get boolean mask of boundary points.

        Args:
            points: Query points - shape (N, d)
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array - True for boundary points
        """
        return self.geometry.is_on_boundary(points, tolerance=tolerance)

    def get_boundary_normals(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Get outward normal vectors at points.

        Only meaningful for points near the boundary.

        Args:
            points: Query points - shape (N, d)

        Returns:
            Normal vectors - shape (N, d)
        """
        return self.geometry.get_boundary_normal(points)

    def get_boundary_regions(self) -> dict[str, dict]:
        """
        Get named boundary regions for mixed BC specification.

        Returns:
            Dictionary mapping region names to region info
        """
        return self.geometry.get_boundary_regions()


class ParticleReflector:
    """
    Optimized particle reflector for Lagrangian FP solvers.

    Handles reflection of particles that cross domain boundaries during
    time integration. Supports both simple projection and physics-based
    elastic reflection.

    Examples:
        >>> reflector = ParticleReflector(domain)
        >>> particles_new = reflector.reflect(particles, velocities, dt)
    """

    def __init__(
        self,
        geometry: GeometryProtocol,
        reflection_type: Literal["simple", "elastic"] = "simple",
    ):
        """
        Initialize particle reflector.

        Args:
            geometry: Domain geometry with boundary methods
            reflection_type: Type of reflection
                - "simple": Project to boundary (no velocity change)
                - "elastic": Reflect velocity at boundary normal
        """
        self.geometry = geometry
        self.reflection_type = reflection_type

    def reflect(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating] | None = None,
        dt: float = 1.0,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """
        Reflect particles at domain boundary.

        Args:
            positions: Particle positions after time step - shape (N, d)
            velocities: Particle velocities - shape (N, d), required for elastic
            dt: Time step (for elastic reflection distance calculation)

        Returns:
            Tuple of (updated_positions, updated_velocities)
        """
        positions = np.atleast_2d(positions)

        if self.reflection_type == "simple":
            # Simple projection back to interior
            new_positions = self.geometry.project_to_interior(positions)
            return new_positions, velocities

        elif self.reflection_type == "elastic":
            if velocities is None:
                raise ValueError("Elastic reflection requires velocities")

            velocities = np.atleast_2d(velocities)
            return self._elastic_reflect(positions, velocities, dt)

        else:
            raise ValueError(f"Unknown reflection type: {self.reflection_type}")

    def _elastic_reflect(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        dt: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Elastic reflection with velocity reversal at boundary normal."""
        new_positions = positions.copy()
        new_velocities = velocities.copy()

        # Find particles at boundary
        on_boundary = self.geometry.is_on_boundary(positions)

        if not np.any(on_boundary):
            return new_positions, new_velocities

        # Project to interior
        new_positions[on_boundary] = self.geometry.project_to_interior(positions[on_boundary])

        # Reflect velocity: v_new = v - 2(v · n)n
        normals = self.geometry.get_boundary_normal(positions[on_boundary])
        v_normal = np.sum(velocities[on_boundary] * normals, axis=1, keepdims=True)
        new_velocities[on_boundary] = velocities[on_boundary] - 2 * v_normal * normals

        return new_positions, new_velocities


__all__ = [
    "MeshfreeApplicator",
    "ParticleReflector",
]
