"""
Adaptive mesh refinement (AMR) for MFG problems.

This module provides AMR support for 1D, 2D, and 3D domains, enabling automatic
grid refinement based on solution gradients and error estimation.

Note: The AMR modules currently use their legacy class names. For new code, use:
  - Interval1D, OneDimensionalAMRMesh, OneDimensionalErrorEstimator (1D)
  - TriangleElement, TriangularAMRMesh, TriangularMeshErrorEstimator (2D triangular)
  - etc.
"""

# Currently empty - AMR classes are imported directly from the old file locations
# in the main geometry/__init__.py for backward compatibility.
# This will be populated when AMR code is migrated to use new unified naming.

__all__ = []
