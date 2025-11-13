"""
Unstructured mesh geometries for MFG problems.

This module provides unstructured mesh support for finite element and finite volume methods,
including mesh generation, refinement, and management.
"""

from .mesh_1d import Mesh1D
from .mesh_2d import Mesh2D
from .mesh_3d import Mesh3D
from .mesh_manager import MeshManager, MeshPipeline

__all__ = [
    "Mesh1D",
    "Mesh2D",
    "Mesh3D",
    "MeshManager",
    "MeshPipeline",
]
