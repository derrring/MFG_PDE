"""
Segment-aware boundary condition applicator for particle methods.

This module provides BC application for Lagrangian particle solvers using
BoundaryConditions segment matching. Unlike MeshfreeApplicator which uses
geometry-based boundary detection, this applicator queries BCSegments to
determine the appropriate action at each boundary point.

BC Type Interpretation for Particles:
- DIRICHLET: Absorb particle (remove from simulation) - exits/sinks
- NEUMANN: Reflect particle (elastic bounce)
- REFLECTING / NO_FLUX: Reflect particle (elastic bounce)
- PERIODIC: Wrap particle to opposite boundary

Architecture:
    BoundaryConditions provides segment matching via get_bc_at_point()
    ParticleApplicator interprets BCType for particle dynamics

Created: 2025-01-03
Part of: Unified Boundary Condition Architecture (Issue #536)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .conditions import BoundaryConditions


class ParticleApplicator:
    """
    Segment-aware boundary condition applicator for particle methods.

    This applicator queries BoundaryConditions to determine the BC type at each
    boundary particle's location, then applies the appropriate action:

    - DIRICHLET: Absorb particle (remove from simulation)
    - NEUMANN: Reflect particle (elastic bounce)
    - REFLECTING / NO_FLUX: Reflect particle (elastic bounce)
    - PERIODIC: Wrap particle to opposite boundary

    Unlike MeshfreeApplicator which uses geometry-based BC, this class uses
    the BoundaryConditions segment matching to support mixed BC (e.g., exits
    on some walls, reflecting on others).

    Examples:
        >>> from mfg_pde.geometry.boundary import mixed_bc, BCSegment, BCType
        >>> bc = mixed_bc(dimension=2, segments=[
        ...     BCSegment("exit", BCType.DIRICHLET, value=0, boundary="right",
        ...               region={"y": (4.0, 6.0)}),
        ...     BCSegment("walls", BCType.REFLECTING, boundary="all", priority=-1),
        ... ])
        >>> applicator = ParticleApplicator()
        >>> remaining, absorbed, exits = applicator.apply(particles, bc, bounds)
    """

    def __init__(self) -> None:
        """Initialize particle applicator."""
        self._boundary_tolerance = 1e-10

    def apply(
        self,
        particles: NDArray[np.floating],
        bc: BoundaryConditions,
        bounds: list[tuple[float, float]],
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating]]:
        """
        Apply segment-aware boundary conditions to particles.

        For each particle that has crossed or is at the boundary:
        1. Query bc.get_bc_at_point() to determine which BC segment applies
        2. Apply the appropriate action based on bc_type

        Args:
            particles: Particle positions, shape (N, d)
            bc: BoundaryConditions with segment definitions (must have domain_bounds set)
            bounds: Domain bounds per dimension [(xmin, xmax), ...] - used if bc.domain_bounds is None

        Returns:
            Tuple of:
            - remaining_particles: Particles not absorbed, shape (M, d) where M <= N
            - absorbed_mask: Boolean mask of absorbed particles, shape (N,)
            - exit_positions: Positions where particles were absorbed, shape (K, d)
        """
        from .types import BCType

        particles = np.atleast_2d(particles)
        n_particles = len(particles)
        dimension = particles.shape[1]

        if n_particles == 0:
            return particles, np.array([], dtype=bool), np.array([]).reshape(0, dimension)

        # Convert bounds to arrays for vectorized operations
        bounds_arr = np.array(bounds)  # Shape: (d, 2)
        domain_min = bounds_arr[:, 0]
        domain_max = bounds_arr[:, 1]
        domain_size = domain_max - domain_min

        # Ensure BC has domain_bounds for segment matching
        if bc.domain_bounds is None:
            bc.domain_bounds = bounds_arr

        # Find particles at or beyond boundary
        at_min = particles <= domain_min + self._boundary_tolerance
        at_max = particles >= domain_max - self._boundary_tolerance
        at_boundary = np.any(at_min | at_max, axis=1)

        # Initialize output arrays
        absorbed_mask = np.zeros(n_particles, dtype=bool)
        exit_positions_list = []
        result_particles = particles.copy()

        # Process boundary particles
        boundary_indices = np.where(at_boundary)[0]

        for idx in boundary_indices:
            particle = result_particles[idx]

            # Determine which boundary the particle is at
            boundary_id = self._get_boundary_id(particle, domain_min, domain_max, dimension)

            # Query BC at this point (uses bc.domain_bounds internally)
            segment = bc.get_bc_at_point(
                particle,
                boundary_id=boundary_id,
            )

            if segment is None:
                # No matching segment - use default reflecting
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)
                continue

            # Apply BC based on type
            if segment.bc_type == BCType.DIRICHLET:
                # Absorbing BC - mark for removal
                absorbed_mask[idx] = True
                exit_positions_list.append(particle.copy())

            elif segment.bc_type in (BCType.REFLECTING, BCType.NO_FLUX, BCType.NEUMANN):
                # Reflecting BC - bounce particle back
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)

            elif segment.bc_type == BCType.PERIODIC:
                # Periodic BC - wrap to opposite boundary
                result_particles[idx] = self._wrap_particle(particle, domain_min, domain_size)

            else:
                # Unknown BC type - default to reflecting
                result_particles[idx] = self._reflect_particle(particle, domain_min, domain_max, domain_size)

        # Build exit positions array
        if exit_positions_list:
            exit_positions = np.array(exit_positions_list)
        else:
            exit_positions = np.array([]).reshape(0, dimension)

        # Filter out absorbed particles
        remaining_particles = result_particles[~absorbed_mask]

        return remaining_particles, absorbed_mask, exit_positions

    def _get_boundary_id(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
        dimension: int,
    ) -> str | None:
        """
        Determine which boundary a particle is at.

        Returns boundary ID like "x_min", "x_max", "y_min", etc.
        """
        axis_names = ["x", "y", "z"] + [f"dim{i}" for i in range(3, dimension)]

        for d in range(dimension):
            if particle[d] <= domain_min[d] + self._boundary_tolerance:
                return f"{axis_names[d]}_min"
            if particle[d] >= domain_max[d] - self._boundary_tolerance:
                return f"{axis_names[d]}_max"

        return None

    def _reflect_particle(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
        domain_size: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Reflect particle back into domain using modular fold reflection.

        Handles particles that may have traveled multiple domain widths.
        """
        result = particle.copy()

        for d in range(len(particle)):
            if domain_size[d] < 1e-14:
                continue

            # Modular fold reflection
            shifted = result[d] - domain_min[d]
            period = 2 * domain_size[d]
            pos_in_period = shifted % period

            if pos_in_period > domain_size[d]:
                pos_in_period = period - pos_in_period

            result[d] = domain_min[d] + pos_in_period

        return result

    def _wrap_particle(
        self,
        particle: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_size: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Wrap particle to opposite boundary (periodic BC)."""
        result = particle.copy()

        for d in range(len(particle)):
            if domain_size[d] < 1e-14:
                continue

            result[d] = domain_min[d] + (particle[d] - domain_min[d]) % domain_size[d]

        return result


__all__ = ["ParticleApplicator"]
