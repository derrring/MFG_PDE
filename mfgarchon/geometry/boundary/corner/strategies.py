"""
Corner handling strategies for boundary conditions (Issue #521).

This module defines the corner strategies available for normal-based
boundary handling. Position-based reflection uses implicit diagonal
strategy (see position.py).

Strategies:
    AVERAGE: Sum adjacent face normals, normalize. Produces diagonal normal.
    PRIORITY: Use first face normal (dimension priority). Simplest.
    MOLLIFY: Treat corner as rounded. Normal points radially from corner vertex.

Usage:
    The corner_strategy parameter is used in:
    - DomainGeometry.get_boundary_normal(corner_strategy=...)
    - BoundaryConditions(corner_strategy=...)

    Position-based functions (reflect_positions, wrap_positions) do NOT use
    this parameter - they always produce implicit diagonal reflection.

Reference:
    See Issue #521 for corner handling architecture.
    See docs/development/CORNER_HANDLING_IMPLEMENTATION_STATUS.md for details.

Created: 2026-01-25 (Issue #521)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class CornerStrategy(str, Enum):
    """
    Corner handling strategy for normal-based boundary conditions.

    At corners where multiple boundary faces meet, the outward normal
    is ambiguous. This enum defines the available strategies.
    """

    AVERAGE = "average"
    """
    Average of adjacent face normals (recommended for most cases).

    For a 2D corner where x_min and y_min faces meet:
        n = normalize((-1, 0) + (0, -1)) = normalize((-1, -1)) = (-0.707, -0.707)

    Produces diagonal reflection for velocity-based methods.
    Position-based reflection (fold) implicitly uses this strategy.
    """

    PRIORITY = "priority"
    """
    Use normal of first face found (dimension priority).

    For a 2D corner where x_min and y_min faces meet:
        n = (-1, 0)  # x-dimension has priority

    Simple but may cause asymmetric behavior at corners.
    Legacy behavior, retained for backward compatibility.
    """

    MOLLIFY = "mollify"
    """
    Treat corner as if rounded (smooth transition).

    Normal points radially outward from the corner vertex toward the query point.
    For a point near corner at (0, 0):
        n = normalize(point - corner_vertex)

    Best for SDF-like domains and smooth gradient computation.
    Falls back to AVERAGE if point is exactly at corner vertex.
    """


# Type alias for function signatures
CornerStrategyLiteral = Literal["average", "priority", "mollify"]

# Default strategy
DEFAULT_CORNER_STRATEGY: CornerStrategyLiteral = "average"


def validate_corner_strategy(strategy: str) -> CornerStrategyLiteral:
    """
    Validate and normalize corner strategy string.

    Args:
        strategy: Strategy name (case-insensitive)

    Returns:
        Validated strategy literal

    Raises:
        ValueError: If strategy is not recognized
    """
    normalized = strategy.lower()
    if normalized not in ("average", "priority", "mollify"):
        raise ValueError(f"Unknown corner strategy: {strategy!r}. Valid options: 'average', 'priority', 'mollify'")
    return normalized  # type: ignore[return-value]


__all__ = [
    "CornerStrategy",
    "CornerStrategyLiteral",
    "DEFAULT_CORNER_STRATEGY",
    "validate_corner_strategy",
]
