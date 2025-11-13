"""
Boundary condition management for MFG problems.

This module provides boundary condition specifications for 1D, 2D, and 3D domains,
supporting Dirichlet, Neumann, Robin, periodic, and no-flux conditions.

Note: Boundary condition classes are currently imported directly from the old file
locations in the main geometry/__init__.py for backward compatibility.
"""

# Import the basic 1D boundary conditions
from .bc_1d import BoundaryConditions

# Import boundary manager
from .bc_manager import BoundaryManager, GeometricBoundaryCondition

# Note: 2D and 3D boundary condition classes (BoundaryCondition2D, BoundaryCondition3D, etc.)
# are still imported from the old file locations for now.

__all__ = [
    "BoundaryConditions",
    "BoundaryManager",
    "GeometricBoundaryCondition",
]
