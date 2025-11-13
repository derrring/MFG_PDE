#!/usr/bin/env python3
"""
Geometry Utilities for MFG Problems

This module provides convenient aliases and utilities for geometric operations
commonly used in MFG problems, particularly for obstacle representation and
signed distance functions.

The actual implementations live in mfg_pde.geometry.implicit, but this module
makes them more discoverable as utilities.

Example:
    >>> from mfg_pde.utils.geometry import RectangleObstacle, CircleObstacle
    >>> from mfg_pde.utils.geometry import Union
    >>>
    >>> # Create obstacles
    >>> wall = RectangleObstacle(np.array([[0.4, 0.6], [0.0, 0.5]]))
    >>> pillar = CircleObstacle(center=np.array([0.7, 0.7]), radius=0.1)
    >>>
    >>> # Combine obstacles
    >>> obstacles = Union([wall, pillar])
    >>>
    >>> # Query signed distance
    >>> distance = obstacles.signed_distance(points)
"""

from __future__ import annotations

# Import from geometry.implicit for convenient access
from mfg_pde.geometry.implicit import Hyperrectangle, Hypersphere, ImplicitDomain
from mfg_pde.geometry.implicit.csg_operations import ComplementDomain, DifferenceDomain, IntersectionDomain, UnionDomain

# =============================================================================
# CONVENIENT ALIASES
# =============================================================================

# More intuitive names for common use cases
RectangleObstacle = Hyperrectangle
"""Alias for Hyperrectangle - represents axis-aligned rectangular obstacles."""

CircleObstacle = Hypersphere
"""Alias for Hypersphere - represents circular/spherical obstacles."""

BoxObstacle = Hyperrectangle
"""Alias for Hyperrectangle - emphasizes 3D box interpretation."""

SphereObstacle = Hypersphere
"""Alias for Hypersphere - emphasizes 3D sphere interpretation."""

# CSG operation aliases (shorter names)
Union = UnionDomain
"""Alias for UnionDomain - combine multiple domains."""

Intersection = IntersectionDomain
"""Alias for IntersectionDomain - intersection of domains."""

Difference = DifferenceDomain
"""Alias for DifferenceDomain - subtract one domain from another."""

Complement = ComplementDomain
"""Alias for ComplementDomain - complement of a domain."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_rectangle_obstacle(xmin: float, xmax: float, ymin: float, ymax: float) -> Hyperrectangle:
    """
    Create a 2D rectangular obstacle.

    Args:
        xmin, xmax: X-axis bounds
        ymin, ymax: Y-axis bounds

    Returns:
        Rectangular obstacle

    Example:
        >>> obstacle = create_rectangle_obstacle(0.4, 0.6, 0.2, 0.8)
        >>> is_inside = obstacle.contains(np.array([[0.5, 0.5]]))
    """
    import numpy as np

    return Hyperrectangle(np.array([[xmin, xmax], [ymin, ymax]]))


def create_circle_obstacle(center_x: float, center_y: float, radius: float) -> Hypersphere:
    """
    Create a 2D circular obstacle.

    Args:
        center_x, center_y: Center coordinates
        radius: Circle radius

    Returns:
        Circular obstacle

    Example:
        >>> obstacle = create_circle_obstacle(0.5, 0.5, 0.2)
        >>> distance = obstacle.signed_distance(np.array([[0.6, 0.6]]))
    """
    import numpy as np

    return Hypersphere(center=np.array([center_x, center_y]), radius=radius)


def create_box_obstacle(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> Hyperrectangle:
    """
    Create a 3D box obstacle.

    Args:
        xmin, xmax: X-axis bounds
        ymin, ymax: Y-axis bounds
        zmin, zmax: Z-axis bounds

    Returns:
        Box obstacle

    Example:
        >>> obstacle = create_box_obstacle(0.4, 0.6, 0.2, 0.8, 0.1, 0.9)
    """
    import numpy as np

    return Hyperrectangle(np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]]))


def create_sphere_obstacle(center_x: float, center_y: float, center_z: float, radius: float) -> Hypersphere:
    """
    Create a 3D spherical obstacle.

    Args:
        center_x, center_y, center_z: Center coordinates
        radius: Sphere radius

    Returns:
        Spherical obstacle

    Example:
        >>> obstacle = create_sphere_obstacle(0.5, 0.5, 0.5, 0.2)
    """
    import numpy as np

    return Hypersphere(center=np.array([center_x, center_y, center_z]), radius=radius)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "BoxObstacle",
    "CircleObstacle",
    "Complement",
    "Difference",
    # Primitive shapes
    "Hyperrectangle",
    "Hypersphere",
    # Base classes
    "ImplicitDomain",
    "Intersection",
    # Obstacle aliases (most common usage)
    "RectangleObstacle",
    "SphereObstacle",
    # CSG operations (short, clean names)
    "Union",
    "create_box_obstacle",
    "create_circle_obstacle",
    # Factory functions
    "create_rectangle_obstacle",
    "create_sphere_obstacle",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for geometry utilities."""
    import numpy as np

    print("Testing geometry utilities...")

    # Test 2D obstacles
    rect = create_rectangle_obstacle(0.4, 0.6, 0.2, 0.8)
    circle = create_circle_obstacle(0.5, 0.5, 0.1)
    assert isinstance(rect, Hyperrectangle)
    assert isinstance(circle, Hypersphere)
    print("✓ 2D obstacle creation works")

    # Test 3D obstacles
    box = create_box_obstacle(0.3, 0.7, 0.2, 0.8, 0.1, 0.9)
    sphere = create_sphere_obstacle(0.5, 0.5, 0.5, 0.15)
    assert isinstance(box, Hyperrectangle)
    assert isinstance(sphere, Hypersphere)
    print("✓ 3D obstacle creation works")

    # Test CSG operations
    union = Union([rect, circle])
    intersection = Intersection([rect, circle])
    difference = Difference(rect, circle)
    print("✓ CSG operations work")

    # Test signed distance queries
    test_points = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    distances_rect = rect.signed_distance(test_points)
    distances_circle = circle.signed_distance(test_points)
    assert distances_rect.shape == (3,)
    assert distances_circle.shape == (3,)
    print("✓ Signed distance queries work")

    # Test containment
    inside_rect = rect.contains(np.array([[0.5, 0.5]]))
    outside_rect = rect.contains(np.array([[0.0, 0.0]]))
    assert inside_rect[0] == True  # noqa: E712
    assert outside_rect[0] == False  # noqa: E712
    print("✓ Containment tests work")

    print("\nAll smoke tests passed!")
