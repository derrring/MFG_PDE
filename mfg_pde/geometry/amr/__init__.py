"""
Adaptive mesh refinement (AMR) for MFG problems.

This module provides AMR support for 1D, 2D, and 3D domains, enabling automatic
grid refinement based on solution gradients and error estimation.

Components:
- 1D AMR: Interval-based refinement for 1D problems
- 2D Triangular AMR: Triangle-based refinement for unstructured 2D meshes
- 2D Quadtree AMR: Quadtree-based refinement for structured 2D grids
- 3D Tetrahedral AMR: Tetrahedral refinement for unstructured 3D meshes
"""

# 1D AMR components
from .amr_1d import (
    Interval1D,
    OneDimensionalAMRMesh,
    OneDimensionalErrorEstimator,
    create_1d_amr_mesh,
)

# 2D Quadtree AMR components
from .amr_quadtree_2d import (
    AdaptiveMesh,
    AMRRefinementCriteria,
    BaseErrorEstimator,
    GradientErrorEstimator,
    QuadTreeNode,
    create_amr_mesh,
)

# 3D Tetrahedral AMR components
from .amr_tetrahedral_3d import (
    TetrahedralAMRMesh,
    TetrahedralErrorEstimator,
    TetrahedronElement,
    create_tetrahedral_amr_mesh,
)

# 2D Triangular AMR components
from .amr_triangular_2d import (
    TriangleElement,
    TriangularAMRMesh,
    TriangularMeshErrorEstimator,
    create_triangular_amr_mesh,
)

__all__ = [
    # 1D AMR
    "Interval1D",
    "OneDimensionalAMRMesh",
    "OneDimensionalErrorEstimator",
    "create_1d_amr_mesh",
    # 2D Triangular AMR
    "TriangleElement",
    "TriangularAMRMesh",
    "TriangularMeshErrorEstimator",
    "create_triangular_amr_mesh",
    # 2D Quadtree AMR
    "AMRRefinementCriteria",
    "AdaptiveMesh",
    "BaseErrorEstimator",
    "GradientErrorEstimator",
    "QuadTreeNode",
    "create_amr_mesh",
    # 3D Tetrahedral AMR
    "TetrahedralAMRMesh",
    "TetrahedralErrorEstimator",
    "TetrahedronElement",
    "create_tetrahedral_amr_mesh",
]
