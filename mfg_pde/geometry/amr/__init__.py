"""
Adaptive mesh refinement (AMR) for MFG problems.

.. warning::
    **EXPERIMENTAL MODULE** - Infrastructure exists but solver integration is incomplete.

    Status (2025-12):
    - Data structures: Complete (1D intervals, 2D quadtree/triangular, 3D tetrahedral)
    - GeometryProtocol: Compliant
    - Solver integration: NOT IMPLEMENTED

    The AMR meshes can be created and refined, but HJB/FP solvers do not yet
    support adaptive mesh operations (interpolation between refinements,
    conservative mass transfer, time-stepping coordination).

    Use uniform grids for production work until solver integration is complete.

This module provides AMR support for 1D, 2D, and 3D domains, enabling automatic
grid refinement based on solution gradients and error estimation.

Available classes:
  - Interval1D, OneDimensionalAMRMesh, OneDimensionalErrorEstimator (1D)
  - QuadTreeNode, QuadTreeMesh, QuadTreeErrorEstimator (2D structured)
  - TriangleElement, TriangularAMRMesh, TriangularMeshErrorEstimator (2D triangular)
  - TetrahedralElement, TetrahedralAMRMesh (3D)
  - AMRRefinementCriteria (shared configuration)
"""

# Note: Runtime warning removed to avoid triggering on every package import.
# The experimental status is documented in the module docstring above.

from .amr_1d import (
    AMRRefinementCriteria,
    Interval1D,
    OneDimensionalAMRMesh,
    OneDimensionalErrorEstimator,
    create_1d_amr_mesh,
)
from .amr_quadtree_2d import (
    AdaptiveMesh,
    BaseErrorEstimator,
    GradientErrorEstimator,
    QuadTreeNode,
)

__all__ = [
    # Shared
    "AMRRefinementCriteria",
    "BaseErrorEstimator",
    "GradientErrorEstimator",
    # 1D
    "Interval1D",
    "OneDimensionalAMRMesh",
    "OneDimensionalErrorEstimator",
    "create_1d_amr_mesh",
    # 2D Quadtree
    "QuadTreeNode",
    "AdaptiveMesh",
]
