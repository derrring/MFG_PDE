"""
Corner Handling for Boundary Conditions (Issue #521).

This module provides unified corner handling across all solver types:

Position-based (implicit diagonal reflection):
    >>> from mfg_pde.geometry.boundary.corner import reflect_positions, wrap_positions
    >>> reflected = reflect_positions(particles, bounds)

    Position functions process all dimensions simultaneously, producing
    diagonal reflection at corners. No strategy parameter needed.

Normal-based (explicit strategy selection):
    >>> from mfg_pde.geometry import DomainGeometry
    >>> geom = DomainGeometry(bounds)
    >>> normal = geom.get_boundary_normal(point, corner_strategy="average")

    Normal queries are methods on DomainGeometry because they need access
    to geometry state. Available strategies: "average", "priority", "mollify".

Velocity-based (TODO - Future):
    >>> from mfg_pde.geometry.boundary.corner import reflect_velocity
    >>> v_new = reflect_velocity(position, velocity, bounds, strategy="average")

    Not yet implemented. For Billiard dynamics and hard-sphere collisions.

Detection:
    >>> geom.is_near_corner(points)  # Boolean array
    >>> geom.get_boundary_faces_at_point(point)  # List of (dim, side) tuples

Module Structure:
    corner/
    ├── __init__.py      # This file - unified exports
    ├── position.py      # reflect_positions, wrap_positions, absorb_positions
    └── strategies.py    # CornerStrategy enum, validation

Architecture Notes:
    - Position functions are STATELESS (pure functions on arrays)
    - Normal queries need GEOMETRY STATE (methods on DomainGeometry)
    - This split avoids circular imports while providing unified access

See Also:
    - docs/development/CORNER_HANDLING_IMPLEMENTATION_STATUS.md
    - Issue #521 for corner handling architecture
"""

# =============================================================================
# Position-based functions (stateless, canonical implementation)
# =============================================================================
from .position import absorb_positions, reflect_positions, wrap_positions

# =============================================================================
# Strategy definitions
# =============================================================================
from .strategies import (
    DEFAULT_CORNER_STRATEGY,
    CornerStrategy,
    CornerStrategyLiteral,
    validate_corner_strategy,
)

__all__ = [
    # Position-based (most common usage)
    "reflect_positions",
    "wrap_positions",
    "absorb_positions",
    # Strategy definitions
    "CornerStrategy",
    "CornerStrategyLiteral",
    "DEFAULT_CORNER_STRATEGY",
    "validate_corner_strategy",
]
