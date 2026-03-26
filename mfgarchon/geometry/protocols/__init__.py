"""
Geometry trait protocols for capability-based interface design.

This module defines trait protocols that geometries can implement to advertise
their capabilities. Solvers can check for required traits at runtime using
isinstance() checks.

Architecture:
- **Operator Traits** (operators.py): Laplacian, Gradient, Divergence, Advection (continuous)
- **Graph Traits** (graph.py): Graph Laplacian, Adjacency, Spatial Embedding (discrete)
- **Topology Traits** (topology.py): Manifold, Lipschitz, Periodic
- **Region Traits** (regions.py): Boundary marking and query capabilities

Design Pattern:
    # Continuous geometry
    class MyGrid(GeometryProtocol, SupportsLaplacian, SupportsGradient):
        def get_laplacian_operator(self, ...):
            # Implementation for continuous Laplacian Δ
            ...

    # Discrete geometry (graph)
    class MyNetwork(GeometryProtocol, SupportsGraphLaplacian, SupportsAdjacency):
        def get_graph_laplacian_operator(self, ...):
            # Implementation for graph Laplacian L = D - A
            ...

    def solve_poisson(geometry: GeometryProtocol):
        if isinstance(geometry, SupportsLaplacian):
            laplacian = geometry.get_laplacian_operator()
        else:
            raise TypeError(f"{type(geometry).__name__} doesn't support Laplacian")

Created: 2026-01-17 (Issue #590 - Phase 1.1, extended in Phase 1.3)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from .graph import (
    SupportsAdjacency,
    SupportsGraphDistance,
    SupportsGraphLaplacian,
    SupportsSpatialEmbedding,
)
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
    # Operator traits (continuous geometries)
    "SupportsLaplacian",
    "SupportsGradient",
    "SupportsDivergence",
    "SupportsAdvection",
    "SupportsInterpolation",
    # Graph traits (discrete geometries)
    "SupportsGraphLaplacian",
    "SupportsAdjacency",
    "SupportsSpatialEmbedding",
    "SupportsGraphDistance",
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
