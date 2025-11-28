"""
Boundary condition types and segments (dimension-agnostic).

This module provides core type definitions for boundary conditions:
- BCType: Enum for boundary condition types
- BCSegment: Specification for a geometric boundary segment

These types are used by the BoundaryConditions class in conditions.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def _compute_sdf_gradient(
    point: np.ndarray,
    sdf_func: Callable[[np.ndarray], float],
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute SDF gradient using central finite differences.

    The gradient of an SDF points in the direction of steepest increase,
    which is the outward normal direction at the boundary.

    Args:
        point: Point at which to evaluate gradient (1D array)
        sdf_func: SDF function
        epsilon: Finite difference step size

    Returns:
        Gradient vector (same shape as point)
    """
    point = np.asarray(point, dtype=float)
    d = len(point)
    grad = np.zeros(d)

    for i in range(d):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += epsilon
        point_minus[i] -= epsilon

        grad[i] = (sdf_func(point_plus) - sdf_func(point_minus)) / (2 * epsilon)

    return grad


class BCType(Enum):
    """Boundary condition types (dimension-agnostic)."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    REFLECTING = "reflecting"  # Alias for Neumann with zero gradient
    NO_FLUX = "no_flux"  # No-flux for Fokker-Planck (same as reflecting)


@dataclass
class BCSegment:
    """
    Boundary condition specification for a geometric segment (dimension-agnostic).

    Defines a boundary condition that applies to a specific region of the domain
    boundary. Works for any spatial dimension using multiple matching methods:

    **Rectangular domains:**
    - `boundary`: String identifier for boundary location (e.g., "left", "right")
    - `region`: Dict mapping axis names/indices to coordinate ranges

    **General domains (SDF-based):**
    - `sdf_region`: Callable SDF defining the segment region (negative = inside)
    - `normal_direction`: Match by outward normal direction
    - `normal_tolerance`: Cosine threshold for normal matching

    Attributes:
        name: Human-readable identifier for this segment
        bc_type: Type of boundary condition (Dirichlet, Neumann, etc.)
        value: BC value (constant or function of (x, t))
        boundary: Boundary location identifier (e.g., "left", "right", "x_min", "x_max")
        region: Axis-specific ranges defining segment extent
        sdf_region: SDF function defining segment region (phi < 0 means in segment)
        normal_direction: Target outward normal for matching (unit vector)
        normal_tolerance: Cosine of max angle between actual and target normal (0.7 = ~45 deg)
        priority: Priority for overlapping segments (higher overrides lower)

    Examples:
        1D segment (rectangular):
        >>> bc = BCSegment(name="left_wall", bc_type=BCType.DIRICHLET, value=0.0, boundary="left")

        2D exit segment on rectangular domain (right wall, y in [4.25, 5.75]):
        >>> exit_bc = BCSegment(
        ...     name="exit",
        ...     bc_type=BCType.DIRICHLET,
        ...     value=0.0,
        ...     boundary="right",
        ...     region={"y": (4.25, 5.75)}
        ... )

        2D exit segment on circular domain (top, normal pointing up):
        >>> exit_bc = BCSegment(
        ...     name="exit",
        ...     bc_type=BCType.DIRICHLET,
        ...     value=0.0,
        ...     normal_direction=np.array([0.0, 1.0]),
        ...     normal_tolerance=0.7,  # cos(45 deg)
        ... )

        SDF-based region (small ball around a point):
        >>> corner_bc = BCSegment(
        ...     name="corner",
        ...     bc_type=BCType.NEUMANN,
        ...     value=0.0,
        ...     sdf_region=lambda x: np.linalg.norm(x - np.array([1, 1])) - 0.2,
        ...     priority=2,
        ... )
    """

    name: str
    bc_type: BCType
    value: float | Callable = 0.0

    # Robin BC coefficients: alpha*u + beta*du/dn = g
    alpha: float = 1.0  # Weight on u (Dirichlet term)
    beta: float = 0.0  # Weight on du/dn (Neumann term)

    # Rectangular domain matching
    boundary: str | None = None
    region: dict[str | int, tuple[float, float]] | None = None

    # SDF-based matching (general domains)
    sdf_region: Callable[[np.ndarray], float] | None = None
    normal_direction: np.ndarray | None = None
    normal_tolerance: float = 0.7  # cos(~45 deg)

    priority: int = 0

    def matches_point(
        self,
        point: np.ndarray,
        boundary_id: str | None,
        domain_bounds: np.ndarray | None,
        tolerance: float = 1e-8,
        axis_names: dict[int, str] | None = None,
        domain_sdf: Callable[[np.ndarray], float] | None = None,
    ) -> bool:
        """
        Check if this BC segment applies to a given boundary point.

        Supports multiple matching modes:
        1. Boundary ID matching (rectangular domains)
        2. Coordinate range matching (rectangular domains)
        3. SDF region matching (general domains)
        4. Normal direction matching (general domains)

        Args:
            point: Spatial coordinates as 1D array of shape (dimension,)
            boundary_id: Boundary identifier (e.g., "left", "right", "x_min", "top")
                        Can be None for SDF-based domains.
            domain_bounds: Domain bounds as array of shape (dimension, 2)
                          bounds[i, 0] = min, bounds[i, 1] = max for axis i
                          Can be None for SDF-based domains.
            tolerance: Tolerance for geometric comparisons
            axis_names: Optional mapping from axis index to name (e.g., {0: "x", 1: "y"})
            domain_sdf: Optional domain SDF for normal-based matching

        Returns:
            True if this BC segment applies to the given point
        """
        # Method 1: Check boundary identifier match (rectangular domains)
        if self.boundary is not None and self.boundary != "all":
            if boundary_id is None or self.boundary != boundary_id:
                return False

        # Method 2: Check coordinate range constraints (rectangular domains)
        if self.region is not None:
            if axis_names is None:
                # Default axis names: x, y, z, w, ...
                axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}

            for axis_key, (range_min, range_max) in self.region.items():
                # Convert axis key to index
                if isinstance(axis_key, str):
                    # Find index from axis name
                    axis_idx = next((idx for idx, name in axis_names.items() if name == axis_key), None)
                    if axis_idx is None:
                        raise ValueError(f"Unknown axis name: {axis_key}")
                else:
                    axis_idx = int(axis_key)

                # Check if point coordinate is within range
                if axis_idx >= len(point):
                    raise ValueError(f"Axis index {axis_idx} exceeds point dimension {len(point)}")

                coord = point[axis_idx]
                if coord < range_min - tolerance or coord > range_max + tolerance:
                    return False

        # Method 3: Check SDF region constraint (general domains)
        if self.sdf_region is not None:
            sdf_val = self.sdf_region(point)
            if sdf_val > tolerance:
                return False

        # Method 4: Check normal direction match (general domains)
        if self.normal_direction is not None and domain_sdf is not None:
            # Compute outward normal via SDF gradient (finite differences)
            normal = _compute_sdf_gradient(point, domain_sdf, epsilon=1e-5)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normal = normal / normal_norm

                # Normalize target direction
                target = np.asarray(self.normal_direction, dtype=float)
                target = target / (np.linalg.norm(target) + 1e-12)

                # Check cosine similarity
                cos_angle = np.dot(normal, target)
                if cos_angle < self.normal_tolerance:
                    return False

        return True

    def get_value(self, point: np.ndarray, t: float = 0.0) -> float:
        """
        Evaluate BC value at a point.

        Args:
            point: Spatial coordinates as 1D array
            t: Time

        Returns:
            BC value at this point and time
        """
        if callable(self.value):
            # Call with (*point, t) for compatibility
            if len(point) == 1:
                return self.value(point[0], t)
            elif len(point) == 2:
                return self.value(point[0], point[1], t)
            elif len(point) == 3:
                return self.value(point[0], point[1], point[2], t)
            else:
                return self.value(point, t)
        return float(self.value)

    def __str__(self) -> str:
        """String representation."""
        parts = [f"{self.name}: {self.bc_type.value}"]

        if self.boundary is not None:
            parts.append(f"on {self.boundary}")

        if self.region is not None:
            region_strs = [f"{axis}in{rng}" for axis, rng in self.region.items()]
            parts.append(f"where {', '.join(region_strs)}")

        if self.sdf_region is not None:
            parts.append("[sdf_region]")

        if self.normal_direction is not None:
            dir_str = ",".join(f"{x:.2f}" for x in self.normal_direction)
            parts.append(f"[normal=({dir_str})]")

        if self.priority > 0:
            parts.append(f"(priority={self.priority})")

        return " ".join(parts)


def create_standard_boundary_names(dimension: int) -> dict[str, str]:
    """
    Create standard boundary name mappings for rectangular domains.

    Args:
        dimension: Spatial dimension

    Returns:
        Dictionary mapping standard names to axis-specific names
    """
    if dimension == 1:
        return {"left": "x_min", "right": "x_max"}
    elif dimension == 2:
        return {
            "left": "x_min",
            "right": "x_max",
            "bottom": "y_min",
            "top": "y_max",
        }
    elif dimension == 3:
        return {
            "left": "x_min",
            "right": "x_max",
            "bottom": "y_min",
            "top": "y_max",
            "front": "z_min",
            "back": "z_max",
        }
    else:
        # nD - use generic names (x, y, z for first 3, then dim3, dim4, ...)
        names = {}
        for i in range(dimension):
            if i < 3:
                axis = ["x", "y", "z"][i]
            else:
                axis = f"dim{i}"
            names[f"{axis}_min"] = f"{axis}_min"
            names[f"{axis}_max"] = f"{axis}_max"
        return names
