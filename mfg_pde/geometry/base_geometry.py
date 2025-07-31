"""
Base geometry classes for MFG_PDE mesh generation pipeline.

This module defines the abstract base classes and core data structures
for the Gmsh → Meshio → PyVista mesh generation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


@dataclass
class MeshData:
    """
    Universal mesh data container compatible with Gmsh → Meshio → PyVista pipeline.

    This class serves as the central data structure for mesh information,
    providing seamless conversion between different mesh formats.

    Attributes:
        vertices: (N_vertices, dim) coordinates [2D/3D compatible]
        elements: Element connectivity (varies by element type)
        element_type: Type of elements ("triangle", "tetrahedron", etc.)
        boundary_tags: Boundary region identifiers
        element_tags: Element region identifiers
        boundary_faces: Boundary face connectivity (2D: edges, 3D: faces)
        dimension: Spatial dimension (2 or 3)
        quality_metrics: Mesh quality analysis data
        element_volumes: Element areas (2D) or volumes (3D)
        metadata: Additional mesh information and parameters
    """

    vertices: np.ndarray  # (N_vertices, dim) coordinates
    elements: np.ndarray  # Element connectivity
    element_type: str  # "triangle", "tetrahedron", etc.
    boundary_tags: np.ndarray  # Boundary region IDs
    element_tags: np.ndarray  # Element region IDs
    boundary_faces: np.ndarray  # Boundary connectivity
    dimension: int  # 2 or 3
    quality_metrics: Dict[str, Any] = None
    element_volumes: np.ndarray = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize derived quantities and validate mesh data."""
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.metadata is None:
            self.metadata = {}

        # Validate dimensions
        if self.dimension not in [2, 3]:
            raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")

        if self.vertices.shape[1] != self.dimension:
            raise ValueError(
                f"Vertex coordinates dimension {self.vertices.shape[1]} "
                f"doesn't match specified dimension {self.dimension}"
            )

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[0]

    @property
    def num_elements(self) -> int:
        """Number of elements in the mesh."""
        return self.elements.shape[0]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box of the mesh: (min_coords, max_coords)."""
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def compute_element_volumes(self) -> np.ndarray:
        """Compute element areas (2D) or volumes (3D)."""
        if self.element_volumes is not None:
            return self.element_volumes

        if self.element_type == "triangle" and self.dimension == 2:
            self.element_volumes = self._compute_triangle_areas()
        elif self.element_type == "tetrahedron" and self.dimension == 3:
            self.element_volumes = self._compute_tetrahedron_volumes()
        else:
            raise NotImplementedError(
                f"Volume computation not implemented for {self.element_type} "
                f"elements in {self.dimension}D"
            )

        return self.element_volumes

    def _compute_triangle_areas(self) -> np.ndarray:
        """Compute areas of triangular elements."""
        areas = np.zeros(self.num_elements)
        for i, element in enumerate(self.elements):
            v0, v1, v2 = self.vertices[element]
            # Area = 0.5 * |cross product|
            cross = np.cross(v1 - v0, v2 - v0)
            areas[i] = 0.5 * abs(cross)
        return areas

    def _compute_tetrahedron_volumes(self) -> np.ndarray:
        """Compute volumes of tetrahedral elements."""
        volumes = np.zeros(self.num_elements)
        for i, element in enumerate(self.elements):
            v0, v1, v2, v3 = self.vertices[element]
            # Volume = (1/6) * |det(v1-v0, v2-v0, v3-v0)|
            matrix = np.array([v1 - v0, v2 - v0, v3 - v0]).T
            volumes[i] = abs(np.linalg.det(matrix)) / 6.0
        return volumes

    def to_meshio(self):
        """Convert to meshio format for I/O operations."""
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio is required for mesh I/O operations")

        cells = [(self.element_type, self.elements)]

        cell_data = {}
        if len(self.element_tags) > 0:
            cell_data[self.element_type] = {"region": self.element_tags}

        point_data = {}
        if len(self.boundary_tags) > 0:
            point_data["boundary"] = self.boundary_tags

        return meshio.Mesh(
            points=self.vertices,
            cells=cells,
            cell_data=cell_data,
            point_data=point_data,
        )

    def to_pyvista(self):
        """Convert to PyVista format for visualization."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista is required for mesh visualization")

        if self.element_type == "triangle":
            # Create PyVista PolyData for 2D triangular mesh
            faces = np.hstack(
                [
                    np.full((self.num_elements, 1), 3),  # 3 vertices per triangle
                    self.elements,
                ]
            )
            mesh = pv.PolyData(self.vertices, faces)

        elif self.element_type == "tetrahedron":
            # Create PyVista UnstructuredGrid for 3D tetrahedral mesh
            cells = np.hstack(
                [
                    np.full((self.num_elements, 1), 4),  # 4 vertices per tetrahedron
                    self.elements,
                ]
            )
            cell_types = np.full(self.num_elements, pv.CellType.TETRA)
            mesh = pv.UnstructuredGrid(cells, cell_types, self.vertices)

        else:
            raise NotImplementedError(
                f"PyVista conversion not implemented for {self.element_type} elements"
            )

        # Add data arrays
        if len(self.boundary_tags) > 0:
            mesh.point_data["boundary_tags"] = self.boundary_tags
        if len(self.element_tags) > 0:
            mesh.cell_data["element_tags"] = self.element_tags

        return mesh


class BaseGeometry(ABC):
    """
    Abstract base class for geometry domains in MFG_PDE.

    This class defines the interface for all geometry types,
    ensuring compatibility with the Gmsh → Meshio → PyVista pipeline.
    """

    def __init__(self, dimension: int):
        """Initialize base geometry.

        Args:
            dimension: Spatial dimension (2 or 3)
        """
        self.dimension = dimension
        self.mesh_data: Optional[MeshData] = None
        self._gmsh_model = None

    @abstractmethod
    def create_gmsh_geometry(self) -> Any:
        """Create geometry using Gmsh API. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def set_mesh_parameters(self, **kwargs) -> None:
        """Set mesh generation parameters. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def generate_mesh(self) -> MeshData:
        """Generate mesh using Gmsh → Meshio pipeline. Must be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get geometry bounding box. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def export_mesh(self, format: str, filename: str) -> None:
        """Export mesh in specified format. Must be implemented by subclasses."""
        pass

    def visualize_mesh(self, show_edges: bool = True, show_quality: bool = False):
        """Visualize mesh using PyVista."""
        if self.mesh_data is None:
            self.generate_mesh()

        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista is required for mesh visualization")

        mesh = self.mesh_data.to_pyvista()
        plotter = pv.Plotter()

        if show_quality:
            # Color by mesh quality if available
            if "quality" in self.mesh_data.quality_metrics:
                mesh.cell_data["quality"] = self.mesh_data.quality_metrics["quality"]
                plotter.add_mesh(mesh, scalars="quality", show_edges=show_edges)
            else:
                plotter.add_mesh(mesh, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, show_edges=show_edges)

        plotter.show()

    def compute_mesh_quality(self) -> Dict[str, float]:
        """Compute mesh quality metrics."""
        if self.mesh_data is None:
            self.generate_mesh()

        quality_metrics = {}

        if self.mesh_data.element_type == "triangle":
            quality_metrics.update(self._compute_triangle_quality())
        elif self.mesh_data.element_type == "tetrahedron":
            quality_metrics.update(self._compute_tetrahedron_quality())

        self.mesh_data.quality_metrics.update(quality_metrics)
        return quality_metrics

    def _compute_triangle_quality(self) -> Dict[str, float]:
        """Compute quality metrics for triangular elements."""
        areas = self.mesh_data.compute_element_volumes()

        # Compute aspect ratios
        aspect_ratios = []
        for i, element in enumerate(self.mesh_data.elements):
            v0, v1, v2 = self.mesh_data.vertices[element]

            # Edge lengths
            e1 = np.linalg.norm(v1 - v0)
            e2 = np.linalg.norm(v2 - v1)
            e3 = np.linalg.norm(v0 - v2)

            # Aspect ratio = (longest edge) / (shortest edge)
            aspect_ratios.append(max(e1, e2, e3) / min(e1, e2, e3))

        return {
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "mean_area": np.mean(areas),
            "min_aspect_ratio": np.min(aspect_ratios),
            "max_aspect_ratio": np.max(aspect_ratios),
            "mean_aspect_ratio": np.mean(aspect_ratios),
        }

    def _compute_tetrahedron_quality(self) -> Dict[str, float]:
        """Compute quality metrics for tetrahedral elements."""
        volumes = self.mesh_data.compute_element_volumes()

        # Basic volume statistics
        return {
            "min_volume": np.min(volumes),
            "max_volume": np.max(volumes),
            "mean_volume": np.mean(volumes),
        }
