"""
Unstructured mesh geometries for MFG problems.

This module provides unstructured mesh support for finite element and finite volume methods,
including mesh generation, refinement, and management.

Note: Uses lazy imports (PEP 562) for Mesh1D/2D/3D to avoid circular import with base.py.
The cycle is: amr_1d -> base -> meshes/__init__ -> mesh_1d -> base
"""

from typing import TYPE_CHECKING

# Eager imports (no cycle risk - mesh_data doesn't import from base.py)
from .mesh_data import MeshData, MeshVisualizationMode

__all__ = [
    "Mesh1D",
    "Mesh2D",
    "Mesh3D",
    "MeshData",
    "MeshManager",
    "MeshPipeline",
    "MeshVisualizationMode",
]

# TYPE_CHECKING block for IDE support (autocomplete, type hints)
# These imports only run during static analysis, not at runtime
if TYPE_CHECKING:
    from .mesh_1d import Mesh1D as Mesh1D
    from .mesh_2d import Mesh2D as Mesh2D
    from .mesh_3d import Mesh3D as Mesh3D
    from .mesh_manager import MeshManager as MeshManager
    from .mesh_manager import MeshPipeline as MeshPipeline


def __getattr__(name: str):
    """Lazy import for mesh classes to break circular import (PEP 562)."""
    if name == "Mesh1D":
        from .mesh_1d import Mesh1D

        return Mesh1D
    elif name == "Mesh2D":
        from .mesh_2d import Mesh2D

        return Mesh2D
    elif name == "Mesh3D":
        from .mesh_3d import Mesh3D

        return Mesh3D
    elif name == "MeshManager":
        from .mesh_manager import MeshManager

        return MeshManager
    elif name == "MeshPipeline":
        from .mesh_manager import MeshPipeline

        return MeshPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
