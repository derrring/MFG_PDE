"""
Geometry trait protocols for capability-based interface design.

This module defines trait protocols that geometries can implement to advertise
their capabilities. Solvers can check for required traits at runtime using
isinstance() checks.

Architecture:
- **Operator Traits** (operators.py): Laplacian, Gradient, Divergence, Advection
- **Topology Traits** (topology.py): Manifold, Lipschitz, Periodic
- **Region Traits** (regions.py): Boundary marking and query capabilities

Design Pattern:
    class MyGeometry(GeometryProtocol, SupportsLaplacian, SupportsGradient):
        def get_laplacian_operator(self, ...):
            # Implementation
            ...

    def solve_poisson(geometry: GeometryProtocol):
        if isinstance(geometry, SupportsLaplacian):
            laplacian = geometry.get_laplacian_operator()
        else:
            raise TypeError(f"{type(geometry).__name__} doesn't support Laplacian")

Created: 2026-01-17 (Issue #590 - Phase 1.1)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from .operators import (
    SupportsAdvection,
    SupportsDivergence,
    SupportsGradient,
    SupportsInterpolation,
    SupportsLaplacian,
)
from .regions import (
    SupportsBoundaryDistance,
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsRegionMarking,
)
from .topology import (
    SupportsLipschitz,
    SupportsManifold,
    SupportsPeriodic,
)

__all__ = [
    # Operator traits
    "SupportsLaplacian",
    "SupportsGradient",
    "SupportsDivergence",
    "SupportsAdvection",
    "SupportsInterpolation",
    # Region traits
    "SupportsBoundaryNormal",
    "SupportsBoundaryProjection",
    "SupportsBoundaryDistance",
    "SupportsRegionMarking",
    # Topology traits
    "SupportsManifold",
    "SupportsLipschitz",
    "SupportsPeriodic",
]
