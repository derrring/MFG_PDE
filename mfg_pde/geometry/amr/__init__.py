"""
Adaptive mesh refinement (AMR) for MFG problems.

.. warning::
    **EXPERIMENTAL MODULE** - Infrastructure exists but solver integration is incomplete.

    Status (2025-12):
    - Data structures: Complete (1D intervals, 2D quadtree/triangular, 3D tetrahedral)
    - GeometryProtocol: Compliant
    - Solver integration: NOT IMPLEMENTED

    The AMR grids can be created and refined, but HJB/FP solvers do not yet
    support adaptive mesh operations (interpolation between refinements,
    conservative mass transfer, time-stepping coordination).

    Use uniform grids for production work until solver integration is complete.

This module provides AMR support for 1D, 2D, and 3D domains, enabling automatic
grid refinement based on solution gradients and error estimation.

Inheritance Hierarchy:
    All AMR classes inherit directly from Geometry (base ABC), NOT from
    CartesianGrid or UnstructuredMesh. This is because AMR classes refine
    existing partitions with dynamic, non-uniform spacing rather than creating
    grids/meshes with predetermined structure.

    Geometry (base ABC)
    ├── CartesianGrid (uniform structured grids - TensorProductGrid)
    ├── UnstructuredMesh (FEM meshes created via Gmsh)
    └── AMR classes (refine existing partitions):
        ├── OneDimensionalAMRGrid (1D hierarchical intervals)
        ├── QuadTreeAMRGrid (2D hierarchical quadrants)
        ├── TriangularAMRMesh (2D triangular refinement)
        └── TetrahedralAMRMesh (3D tetrahedral refinement)

    Design rationale:
    - CartesianGrid requires uniform spacing (dx, dy) - AMR has variable spacing
    - UnstructuredMesh requires Gmsh mesh generation - AMR adapts existing meshes
    - Future: Consider AdaptiveGeometry ABC if common AMR interface emerges

Available classes:
  - Interval1D, OneDimensionalAMRGrid, OneDimensionalErrorEstimator (1D)
  - QuadTreeNode, QuadTreeAMRGrid, GradientErrorEstimator (2D structured)
  - TriangleElement, TriangularAMRMesh, TriangularMeshErrorEstimator (2D triangular)
  - TetrahedronElement, TetrahedralAMRMesh, TetrahedralErrorEstimator (3D)
  - AMRRefinementCriteria (shared configuration)
"""

# Note: Runtime warning removed to avoid triggering on every package import.
# The experimental status is documented in the module docstring above.

from .amr_1d import (
    AMRRefinementCriteria,
    Interval1D,
    # New names (v0.16.6+)
    OneDimensionalAMRGrid,
    # Deprecated aliases (will be removed in v1.0.0)
    OneDimensionalAMRMesh,
    OneDimensionalErrorEstimator,
    create_1d_amr_grid,
    create_1d_amr_mesh,
)
from .amr_quadtree_2d import (
    # Deprecated aliases (will be removed in v1.0.0)
    AdaptiveMesh,
    # Error estimators
    BaseErrorEstimator,
    GradientErrorEstimator,
    # New names (v0.16.6+)
    QuadTreeAMRGrid,
    QuadTreeNode,
    create_amr_mesh,
    create_quadtree_amr_grid,
)
from .amr_tetrahedral_3d import (
    TetrahedralAMRMesh,
    TetrahedralErrorEstimator,
    TetrahedronElement,
)
from .amr_triangular_2d import (
    TriangleElement,
    TriangularAMRMesh,
    TriangularMeshErrorEstimator,
)

__all__ = [
    # Shared
    "AMRRefinementCriteria",
    "BaseErrorEstimator",
    "GradientErrorEstimator",
    # 1D (new names)
    "Interval1D",
    "OneDimensionalAMRGrid",
    "OneDimensionalErrorEstimator",
    "create_1d_amr_grid",
    # 1D (deprecated aliases)
    "OneDimensionalAMRMesh",
    "create_1d_amr_mesh",
    # 2D Quadtree (new names)
    "QuadTreeNode",
    "QuadTreeAMRGrid",
    "create_quadtree_amr_grid",
    # 2D Quadtree (deprecated aliases)
    "AdaptiveMesh",
    "create_amr_mesh",
    # 2D Triangular
    "TriangleElement",
    "TriangularAMRMesh",
    "TriangularMeshErrorEstimator",
    # 3D Tetrahedral
    "TetrahedronElement",
    "TetrahedralAMRMesh",
    "TetrahedralErrorEstimator",
]
