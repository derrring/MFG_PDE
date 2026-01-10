"""
Unified boundary condition specification (dimension-agnostic).

This module provides the canonical BoundaryConditions class supporting:
- **Uniform BCs**: Single segment covering all boundaries (same type everywhere)
- **Mixed BCs**: Multiple segments with different types on different boundaries
- **Rectangular domains**: Axis-aligned boundaries via `domain_bounds`
- **General/Lipschitz domains**: SDF-defined boundaries via `domain_sdf`

Use factory functions for convenient creation:
- `uniform_bc()`, `periodic_bc()`, `dirichlet_bc()`, etc. for uniform BCs
- `mixed_bc()` for mixed BCs with multiple segments

Examples:
    Uniform Neumann BC:
    >>> bc = neumann_bc(dimension=2)
    >>> assert bc.is_uniform

    Mixed BC with exit and walls:
    >>> from mfg_pde.geometry.boundary import BCSegment, BCType
    >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
    ...                      boundary="x_max", priority=1)
    >>> wall_seg = BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0)
    >>> bc = mixed_bc([exit_seg, wall_seg], dimension=2,
    ...               domain_bounds=np.array([[0, 10], [0, 10]]))
    >>> assert bc.is_mixed
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from .types import BCSegment, BCType, _compute_sdf_gradient

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BoundaryConditions:
    """
    Unified boundary condition specification (uniform or mixed, any dimension).

    This is the canonical boundary condition class supporting:
    - **Uniform BCs**: Single segment covering all boundaries (same type everywhere)
    - **Mixed BCs**: Multiple segments with different types on different boundaries
    - **Rectangular domains**: Axis-aligned boundaries via `domain_bounds`
    - **General/Lipschitz domains**: SDF-defined boundaries via `domain_sdf`

    Use factory functions for convenient creation:
    - `uniform_bc()`, `periodic_bc()`, `dirichlet_bc()`, etc. for uniform BCs
    - `mixed_bc()` for mixed BCs with multiple segments

    Attributes:
        dimension: Spatial dimension of the problem (1, 2, 3, ...) or None for lazy binding.
            When None, dimension will be inferred when BC is attached to a Geometry.
        segments: List of BC segments (ordered by priority)
        default_bc: Default BC type when no segment matches
        default_value: Default BC value when no segment matches
        domain_bounds: Domain bounds array of shape (dimension, 2) for rectangular domains
        domain_sdf: Signed distance function for general/Lipschitz domains
        corner_strategy: How to handle corners/edges ("priority", "average", "mollify")
        corner_mollification_radius: Smoothing radius for "mollify" strategy

    Examples:
        Uniform Neumann BC:
        >>> bc = neumann_bc(dimension=2)
        >>> assert bc.is_uniform

        Mixed BC with exit and walls:
        >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
        ...                      boundary="x_max", priority=1)
        >>> wall_seg = BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0)
        >>> bc = mixed_bc([exit_seg, wall_seg], dimension=2,
        ...               domain_bounds=np.array([[0, 10], [0, 10]]))
        >>> assert bc.is_mixed

        Circular domain with exit at top (Lipschitz/SDF):
        >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
        ...                      normal_direction=np.array([0, 1]), priority=1)
        >>> bc = mixed_bc([exit_seg], dimension=2,
        ...               domain_sdf=lambda x: np.linalg.norm(x) - 5.0)
    """

    dimension: int | None = None
    segments: list[BCSegment] = field(default_factory=list)
    default_bc: BCType = BCType.PERIODIC
    default_value: float = 0.0

    # Rectangular domain specification
    domain_bounds: np.ndarray | None = None

    # General domain specification (SDF-based, supports Lipschitz boundaries)
    domain_sdf: Callable[[np.ndarray], float] | None = None

    # Corner handling (important for Lipschitz domains with re-entrant corners)
    corner_strategy: Literal["priority", "average", "mollify"] = "priority"
    corner_mollification_radius: float = 0.1

    def __post_init__(self):
        """Sort segments by priority (highest first)."""
        self.segments.sort(key=lambda seg: seg.priority, reverse=True)

    # =========================================================================
    # Lazy dimension binding
    # =========================================================================

    @property
    def is_bound(self) -> bool:
        """Check if dimension has been bound (explicitly or via lazy binding)."""
        return self.dimension is not None

    def bind_dimension(self, dim: int) -> BoundaryConditions:
        """
        Bind dimension to this BC specification (lazy binding).

        Called by Geometry when BC is attached. If dimension is already set,
        validates consistency. Returns a new BoundaryConditions instance with
        the bound dimension.

        Args:
            dim: Spatial dimension to bind

        Returns:
            BoundaryConditions with dimension set

        Raises:
            ValueError: If BC already has a different dimension

        Example:
            >>> bc = dirichlet_bc(value=0.0)  # dimension=None
            >>> bc_2d = bc.bind_dimension(2)  # Now dimension=2
            >>> assert bc_2d.dimension == 2
        """
        if self.dimension is not None and self.dimension != dim:
            raise ValueError(
                f"BC dimension mismatch: BC has dimension={self.dimension}, but geometry has dimension={dim}"
            )
        if self.dimension == dim:
            return self  # Already bound to correct dimension
        return replace(self, dimension=dim)

    def _require_dimension(self, operation: str = "this operation") -> int:
        """
        Internal helper: require dimension to be bound before certain operations.

        Args:
            operation: Name of operation for error message

        Returns:
            Bound dimension

        Raises:
            ValueError: If dimension is not bound
        """
        if self.dimension is None:
            raise ValueError(
                f"BC dimension not set. Cannot perform {operation}. "
                f"Either specify dimension in factory function (e.g., dirichlet_bc(dimension=2)) "
                f"or attach BC to a Geometry to bind dimension automatically."
            )
        return self.dimension

    # =========================================================================
    # Properties to distinguish uniform vs mixed BCs
    # =========================================================================

    @property
    def is_uniform(self) -> bool:
        """
        Check if this is a uniform BC (single segment covering all boundaries).

        Uniform BCs have exactly one segment with no boundary restriction.
        """
        if len(self.segments) != 1:
            return False
        seg = self.segments[0]
        # Uniform if no specific boundary, region, sdf_region, or normal_direction
        return seg.boundary is None and seg.region is None and seg.sdf_region is None and seg.normal_direction is None

    @property
    def is_mixed(self) -> bool:
        """
        Check if this is a mixed BC (multiple segments or boundary-specific).

        Mixed BCs have multiple segments or segments targeting specific boundaries.
        """
        return not self.is_uniform

    @property
    def type(self) -> str:
        """
        Get the BC type string (for uniform BCs).

        For uniform BCs, returns the type string (e.g., "periodic", "dirichlet").
        For mixed BCs, raises ValueError - use segments directly.

        This property provides compatibility with code expecting the old
        BoundaryConditions.type attribute.
        """
        if not self.is_uniform:
            raise ValueError("type property only valid for uniform BCs. For mixed BCs, access segments directly.")
        return self.segments[0].bc_type.value

    @property
    def bc_type(self) -> BCType:
        """
        Get the BCType enum (for uniform BCs).

        For uniform BCs, returns the BCType enum value.
        For mixed BCs, raises ValueError.
        """
        if not self.is_uniform:
            raise ValueError("bc_type property only valid for uniform BCs. For mixed BCs, access segments directly.")
        return self.segments[0].bc_type

    def get_bc_at_point(
        self,
        point: np.ndarray,
        boundary_id: str | None = None,
        tolerance: float = 1e-8,
        axis_names: dict[int, str] | None = None,
    ) -> BCSegment:
        """
        Get the BC segment that applies to a specific boundary point.

        Args:
            point: Spatial coordinates as 1D array
            boundary_id: Boundary identifier (can be None for SDF-based domains)
            tolerance: Tolerance for geometric comparisons
            axis_names: Optional axis name mapping

        Returns:
            BCSegment that applies (highest priority match, or default)
        """
        # Validate that at least one domain specification is provided
        if self.domain_bounds is None and self.domain_sdf is None:
            raise ValueError("Either domain_bounds or domain_sdf must be set")

        # For SDF domains, auto-identify boundary if not provided
        if boundary_id is None and self.domain_sdf is not None:
            boundary_id = self.identify_boundary_id(point, tolerance)

        # Check segments in priority order (already sorted)
        for segment in self.segments:
            if segment.matches_point(
                point,
                boundary_id,
                self.domain_bounds,
                tolerance,
                axis_names,
                domain_sdf=self.domain_sdf,
            ):
                return segment

        # No match - return default BC as a segment
        return BCSegment(
            name="default",
            bc_type=self.default_bc,
            value=self.default_value,
            priority=-1,
        )

    def get_bc_type_at_boundary(self, boundary: str) -> BCType:
        """
        Get the BC type at a specific boundary (safe accessor for mixed BCs).

        This method provides a safe way to query BC types for both uniform and mixed BCs.
        For solvers that need to know the BC type at a specific boundary (e.g., "x_min",
        "y_max"), this method handles the priority resolution for mixed BCs.

        Args:
            boundary: Boundary identifier (e.g., "x_min", "x_max", "y_min", "y_max")

        Returns:
            BCType at the specified boundary

        Examples:
            >>> bc = neumann_bc(dimension=2)
            >>> bc.get_bc_type_at_boundary("x_min")
            BCType.NEUMANN

            >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max")
            >>> wall_seg = BCSegment(name="wall", bc_type=BCType.NEUMANN)
            >>> bc = mixed_bc([exit_seg, wall_seg], dimension=2, domain_bounds=bounds)
            >>> bc.get_bc_type_at_boundary("x_max")
            BCType.DIRICHLET
            >>> bc.get_bc_type_at_boundary("x_min")
            BCType.NEUMANN
        """
        # For uniform BCs, return the single type
        if self.is_uniform:
            return self.segments[0].bc_type

        # For mixed BCs, find the highest priority segment matching this boundary
        for segment in self.segments:  # Already sorted by priority
            if segment.boundary is None:
                # Default segment matches all boundaries
                return segment.bc_type
            if segment.boundary == boundary:
                return segment.bc_type

        # No match - return default BC type
        return self.default_bc

    def get_bc_value_at_boundary(self, boundary: str, time: float = 0.0, point: np.ndarray | None = None) -> float:
        """
        Get the BC value at a specific boundary (safe accessor for mixed BCs).

        Args:
            boundary: Boundary identifier (e.g., "x_min", "x_max")
            time: Current time for time-dependent BCs
            point: Optional spatial point for spatially-varying BCs

        Returns:
            BC value at the specified boundary
        """
        # For uniform BCs, return the single value
        if self.is_uniform:
            seg = self.segments[0]
            if callable(seg.value):
                if point is not None:
                    return seg.value(point, time)
                return seg.value(time)
            return seg.value

        # For mixed BCs, find the highest priority segment
        for segment in self.segments:
            if segment.boundary is None or segment.boundary == boundary:
                if callable(segment.value):
                    if point is not None:
                        return segment.value(point, time)
                    return segment.value(time)
                return segment.value

        # No match - return default value
        return self.default_value

    def identify_boundary_id(self, point: np.ndarray, tolerance: float = 1e-8) -> str | None:
        """
        Identify which boundary a point lies on.

        For rectangular domains, returns axis-aligned boundary IDs (e.g., "x_min", "y_max").
        For SDF domains, returns normal-based boundary IDs based on the dominant normal direction.

        Args:
            point: Spatial coordinates
            tolerance: Tolerance for boundary detection

        Returns:
            Boundary identifier string or None if not on boundary
        """
        dimension = self._require_dimension("identify_boundary_id")
        point = np.asarray(point, dtype=float)

        # Method 1: Rectangular domain (axis-aligned detection)
        if self.domain_bounds is not None:
            axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}

            for axis_idx in range(dimension):
                axis_name = axis_names.get(axis_idx, f"axis{axis_idx}")

                # Check if on min boundary for this axis
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 0]) < tolerance:
                    return f"{axis_name}_min"

                # Check if on max boundary for this axis
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 1]) < tolerance:
                    return f"{axis_name}_max"

            return None

        # Method 2: SDF domain (normal-based detection)
        if self.domain_sdf is not None:
            # Check if point is on boundary (|phi| < tolerance)
            phi = self.domain_sdf(point)
            if abs(phi) > tolerance:
                return None  # Not on boundary

            # Compute outward normal via SDF gradient
            normal = _compute_sdf_gradient(point, self.domain_sdf, epsilon=1e-5)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                return "boundary"  # Degenerate case

            normal = normal / normal_norm

            # Map normal to boundary ID based on dominant component
            return self._normal_to_boundary_id(normal)

        raise ValueError("Either domain_bounds or domain_sdf must be set")

    def _normal_to_boundary_id(self, normal: np.ndarray) -> str:
        """
        Map outward normal vector to a boundary identifier string.

        Args:
            normal: Unit outward normal vector

        Returns:
            Boundary identifier based on dominant normal direction
        """
        axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}

        # Find axis with largest absolute normal component
        abs_normal = np.abs(normal)
        dominant_axis = int(np.argmax(abs_normal))

        axis_name = axis_names.get(dominant_axis, f"axis{dominant_axis}")

        # Determine direction (min or max)
        if normal[dominant_axis] > 0:
            return f"{axis_name}_max"
        else:
            return f"{axis_name}_min"

    def is_on_boundary(self, point: np.ndarray, tolerance: float = 1e-8) -> bool:
        """
        Check if a point is on the domain boundary.

        Args:
            point: Spatial coordinates
            tolerance: Tolerance for boundary detection

        Returns:
            True if point is on the boundary
        """
        dimension = self._require_dimension("is_on_boundary")
        point = np.asarray(point, dtype=float)

        # Rectangular domain: check if on any axis boundary
        if self.domain_bounds is not None:
            for axis_idx in range(dimension):
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 0]) < tolerance:
                    return True
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 1]) < tolerance:
                    return True
            return False

        # SDF domain: check if |phi| < tolerance
        if self.domain_sdf is not None:
            phi = self.domain_sdf(point)
            return abs(phi) < tolerance

        return False

    def get_outward_normal(self, point: np.ndarray, epsilon: float = 1e-5) -> np.ndarray | None:
        """
        Get the outward normal at a boundary point.

        Args:
            point: Spatial coordinates on the boundary
            epsilon: Finite difference step for SDF gradient

        Returns:
            Unit outward normal vector, or None if not available
        """
        dimension = self._require_dimension("get_outward_normal")
        point = np.asarray(point, dtype=float)

        # SDF domain: use gradient
        if self.domain_sdf is not None:
            normal = _compute_sdf_gradient(point, self.domain_sdf, epsilon=epsilon)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                return normal / normal_norm
            return None

        # Rectangular domain: compute based on boundary ID
        if self.domain_bounds is not None:
            boundary_id = self.identify_boundary_id(point)
            if boundary_id is None:
                return None

            normal = np.zeros(dimension)
            for axis_idx in range(dimension):
                axis_name = ["x", "y", "z", "w"][axis_idx] if axis_idx < 4 else f"axis{axis_idx}"
                if boundary_id == f"{axis_name}_min":
                    normal[axis_idx] = -1.0
                    return normal
                elif boundary_id == f"{axis_name}_max":
                    normal[axis_idx] = 1.0
                    return normal

        return None

    # =========================================================================
    # Flux-Limited Absorption (for DIRICHLET exits)
    # =========================================================================

    def has_flux_limits(self) -> bool:
        """Check if any segment has flux capacity limits."""
        return any(seg.flux_capacity is not None for seg in self.segments)

    def get_flux_limits(self) -> dict[str, float]:
        """
        Get flux capacities for all segments that have limits.

        Returns:
            Dict mapping segment name to flux capacity (mass/time or particles/time).
            Only includes segments with explicit flux_capacity set.

        Example:
            >>> bc = mixed_bc(segments=[
            ...     BCSegment("exit_A", BCType.DIRICHLET, flux_capacity=0.1),
            ...     BCSegment("exit_B", BCType.DIRICHLET, flux_capacity=0.2),
            ...     BCSegment("walls", BCType.NO_FLUX),  # No flux limit
            ... ], dimension=2)
            >>> bc.get_flux_limits()
            {'exit_A': 0.1, 'exit_B': 0.2}
        """
        return {seg.name: seg.flux_capacity for seg in self.segments if seg.flux_capacity is not None}

    def get_flux_limit_for_segment(self, name: str) -> float | None:
        """Get flux capacity for a specific segment by name."""
        for seg in self.segments:
            if seg.name == name:
                return seg.flux_capacity
        return None

    def compute_particle_flux_limits(
        self,
        dt: float,
        n_particles: int,
        total_mass: float = 1.0,
    ) -> dict[str, int]:
        """
        Convert mass-based flux capacities to particle counts for a timestep.

        For particle methods, flux_capacity is in mass/time units.
        This converts to max particles per timestep.

        Args:
            dt: Timestep duration
            n_particles: Total number of particles in simulation
            total_mass: Total mass represented by particles (default 1.0)

        Returns:
            Dict mapping segment name to max particles absorbed per timestep.

        Example:
            >>> # flux_capacity=0.1 means 10% of total mass can exit per unit time
            >>> bc.compute_particle_flux_limits(dt=0.1, n_particles=1000, total_mass=1.0)
            {'exit_A': 10}  # 0.1 * 0.1 * 1000 = 10 particles
        """
        mass_per_particle = total_mass / n_particles
        limits = {}

        for seg in self.segments:
            if seg.flux_capacity is not None:
                # flux_capacity * dt = mass that can exit this timestep
                # Divide by mass_per_particle = max particles
                max_particles = int(seg.flux_capacity * dt / mass_per_particle)
                limits[seg.name] = max(1, max_particles)  # At least 1 if any capacity

        return limits

    def validate_values(self) -> None:
        """
        Validate that required values are provided for boundary condition segments.

        For uniform BCs, checks that the segment has appropriate values set.
        For mixed BCs, validates each segment individually.

        Raises:
            ValueError: If required values are missing for a BC type.

        Note:
            This method provides backward compatibility with the old
            fdm_bc_1d.BoundaryConditions.validate_values() method.
        """
        for segment in self.segments:
            bc_type = segment.bc_type

            if bc_type in (BCType.DIRICHLET, BCType.NEUMANN, BCType.NO_FLUX):
                # These types require a value (can be 0.0 which is valid)
                if segment.value is None:
                    raise ValueError(f"Segment '{segment.name}' with {bc_type.value} BC requires a value")

            elif bc_type == BCType.ROBIN:
                # Robin requires alpha, beta, and value
                if segment.alpha is None or segment.beta is None:
                    raise ValueError(f"Segment '{segment.name}' with Robin BC requires alpha and beta coefficients")
                if segment.value is None:
                    raise ValueError(f"Segment '{segment.name}' with Robin BC requires a value")

            # Periodic BC doesn't require values - it's handled by wrapping

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the mixed BC configuration.

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check that dimension is set for full validation
        if self.dimension is None:
            warnings.append("Dimension not set. Some validation checks skipped.")

        # Check that at least one domain specification exists
        if self.domain_bounds is None and self.domain_sdf is None:
            warnings.append("Neither domain_bounds nor domain_sdf is set")

        # Check segments have valid dimension (for rectangular domains) - only if dimension is set
        if self.dimension is not None:
            for segment in self.segments:
                if segment.region is not None:
                    max_axis = max(
                        (k if isinstance(k, int) else 0 for k in segment.region),
                        default=-1,
                    )
                    if max_axis >= self.dimension:
                        warnings.append(f"Segment '{segment.name}' region exceeds dimension {self.dimension}")

                # Check normal_direction dimension
                if segment.normal_direction is not None:
                    if len(segment.normal_direction) != self.dimension:
                        warnings.append(
                            f"Segment '{segment.name}' normal_direction has wrong dimension: "
                            f"expected {self.dimension}, got {len(segment.normal_direction)}"
                        )

        # Check for conflicting segments with same priority
        priority_groups = {}
        for segment in self.segments:
            if segment.priority not in priority_groups:
                priority_groups[segment.priority] = []
            priority_groups[segment.priority].append(segment)

        for priority, group in priority_groups.items():
            if len(group) > 1:
                warnings.append(f"Multiple segments with priority {priority}: {[s.name for s in group]}")

        # Check boundary coverage for rectangular domains - only if dimension is set
        # Warn if no segments cover certain boundaries (may indicate incomplete BC specification)
        if self.dimension is not None and self.domain_bounds is not None and not self.is_uniform:
            # Map axis index to standard names
            axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}

            for axis_idx in range(self.dimension):
                axis_name = axis_names.get(axis_idx, f"axis{axis_idx}")

                # Check if min and max boundaries on this axis have at least one segment
                # A segment covers a boundary if:
                # 1. boundary is None (applies to all boundaries)
                # 2. boundary explicitly names this boundary (e.g., "x_min", "left")
                # 3. No region constraint (applies to entire boundary)

                min_boundary_names = {f"{axis_name}_min", "left", "bottom", "front"}
                max_boundary_names = {f"{axis_name}_max", "right", "top", "back"}

                has_min_coverage = any(
                    seg.boundary is None
                    or seg.boundary == "all"
                    or (axis_idx == 0 and seg.boundary in min_boundary_names)
                    or (axis_idx == 1 and seg.boundary in min_boundary_names)
                    or (axis_idx == 2 and seg.boundary in min_boundary_names)
                    for seg in self.segments
                )

                has_max_coverage = any(
                    seg.boundary is None
                    or seg.boundary == "all"
                    or (axis_idx == 0 and seg.boundary in max_boundary_names)
                    or (axis_idx == 1 and seg.boundary in max_boundary_names)
                    or (axis_idx == 2 and seg.boundary in max_boundary_names)
                    for seg in self.segments
                )

                # If no explicit segment coverage, default BC will be used - issue info warning
                if not has_min_coverage:
                    warnings.append(
                        f"No explicit BC segment for {axis_name}_min boundary. "
                        f"Default BC ({self.default_bc.value}) will be used."
                    )
                if not has_max_coverage:
                    warnings.append(
                        f"No explicit BC segment for {axis_name}_max boundary. "
                        f"Default BC ({self.default_bc.value}) will be used."
                    )

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def __str__(self) -> str:
        """String representation."""
        dim_str = f"{self.dimension}D" if self.dimension is not None else "unbound"
        if self.is_uniform:
            seg = self.segments[0]
            return f"BoundaryConditions({dim_str}, {seg.bc_type.value}, value={seg.value})"

        # Mixed BC
        domain_type = "rectangular" if self.domain_bounds is not None else "SDF"
        lines = [f"BoundaryConditions({dim_str}, mixed, {domain_type}):"]
        for segment in self.segments:
            lines.append(f"  - {segment}")
        lines.append(f"  - Default: {self.default_bc.value} = {self.default_value}")
        if self.corner_strategy != "priority":
            lines.append(f"  - Corner handling: {self.corner_strategy}")
        return "\n".join(lines)


# =============================================================================
# Factory Functions for Boundary Conditions
# =============================================================================


def uniform_bc(
    bc_type: str | BCType,
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> BoundaryConditions:
    """
    Create uniform boundary conditions (same type on all boundaries).

    Args:
        bc_type: BC type ("periodic", "dirichlet", "neumann", "robin", "no_flux")
        value: BC value (constant or callable(point, time))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds array (dimension, 2)
        alpha: Robin coefficient for u term (only for Robin BC)
        beta: Robin coefficient for du/dn term (only for Robin BC)

    Returns:
        BoundaryConditions with single uniform segment
    """
    if isinstance(bc_type, str):
        bc_type = BCType(bc_type.lower())

    segment = BCSegment(
        name="uniform",
        bc_type=bc_type,
        value=value,
        alpha=alpha,
        beta=beta,
        priority=0,
    )
    return BoundaryConditions(
        dimension=dimension,
        segments=[segment],
        domain_bounds=domain_bounds,
        default_bc=bc_type,
        default_value=value if not callable(value) else 0.0,
    )


def periodic_bc(dimension: int | None = None, domain_bounds: np.ndarray | None = None) -> BoundaryConditions:
    """
    Create periodic boundary conditions.

    Args:
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform periodic BC
    """
    return uniform_bc(BCType.PERIODIC, value=0.0, dimension=dimension, domain_bounds=domain_bounds)


def dirichlet_bc(
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Dirichlet boundary conditions (u = value at boundary).

    Args:
        value: Boundary value (constant or callable(point, time))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Dirichlet BC
    """
    return uniform_bc(BCType.DIRICHLET, value=value, dimension=dimension, domain_bounds=domain_bounds)


def neumann_bc(
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Neumann boundary conditions (du/dn = value at boundary).

    Args:
        value: Normal derivative value (constant or callable(point, time))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Neumann BC
    """
    return uniform_bc(BCType.NEUMANN, value=value, dimension=dimension, domain_bounds=domain_bounds)


def no_flux_bc(dimension: int | None = None, domain_bounds: np.ndarray | None = None) -> BoundaryConditions:
    """
    Create no-flux boundary conditions (zero normal derivative).

    Equivalent to Neumann BC with value=0. Common for Fokker-Planck equations.

    Args:
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform no-flux BC
    """
    return uniform_bc(BCType.NO_FLUX, value=0.0, dimension=dimension, domain_bounds=domain_bounds)


def robin_bc(
    value: float | Callable = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Robin boundary conditions (alpha*u + beta*du/dn = value).

    Args:
        value: RHS value g in alpha*u + beta*du/dn = g
        alpha: Coefficient of u
        beta: Coefficient of du/dn
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Robin BC
    """
    return uniform_bc(
        BCType.ROBIN,
        value=value,
        dimension=dimension,
        domain_bounds=domain_bounds,
        alpha=alpha,
        beta=beta,
    )


def mixed_bc(
    segments: list[BCSegment],
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
    domain_sdf: Callable[[np.ndarray], float] | None = None,
    default_bc: BCType = BCType.NEUMANN,
    default_value: float = 0.0,
    corner_strategy: Literal["priority", "average", "mollify"] = "priority",
) -> BoundaryConditions:
    """
    Create mixed boundary conditions (different types on different segments).

    Supports both rectangular domains (via domain_bounds) and general/Lipschitz
    domains (via domain_sdf with SDF-based boundary detection).

    Args:
        segments: List of BCSegment defining BCs on different boundary parts
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Domain bounds array (dimension, 2) for rectangular domains
        domain_sdf: Signed distance function for general/Lipschitz domains
        default_bc: Default BC type when no segment matches
        default_value: Default BC value when no segment matches
        corner_strategy: How to handle corners ("priority", "average", "mollify")

    Returns:
        Mixed BoundaryConditions

    Example:
        >>> exit = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
        ...                  boundary="x_max", priority=1)
        >>> wall = BCSegment(name="wall", bc_type=BCType.NEUMANN, value=0.0)
        >>> bc = mixed_bc([exit, wall], dimension=2,
        ...               domain_bounds=np.array([[0, 1], [0, 1]]))
    """
    return BoundaryConditions(
        dimension=dimension,
        segments=segments,
        domain_bounds=domain_bounds,
        domain_sdf=domain_sdf,
        default_bc=default_bc,
        default_value=default_value,
        corner_strategy=corner_strategy,
    )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for backward compatibility with code using MixedBoundaryConditions
MixedBoundaryConditions = BoundaryConditions
