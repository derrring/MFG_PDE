"""
Simple grid geometry for high-dimensional MFG problems.

This module provides basic rectangular grid generation without external dependencies,
suitable for regular domain problems and testing.
"""

from __future__ import annotations


import numpy as np

from .base_geometry import BaseGeometry, MeshData


class SimpleGrid2D(BaseGeometry):
    """Simple 2D rectangular grid without external mesh dependencies."""

    def __init__(self,
                 bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                 resolution: tuple[int, int] = (32, 32)):
        """
        Initialize 2D grid.

        Args:
            bounds: (xmin, xmax, ymin, ymax)
            resolution: (nx, ny) grid points
        """
        super().__init__(dimension=2)

        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.nx, self.ny = resolution
        self._bounds = bounds

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get domain bounds."""
        return self._bounds

    def create_gmsh_geometry(self):
        """Not implemented for simple grid - raises NotImplementedError."""
        raise NotImplementedError("SimpleGrid2D does not use Gmsh")

    def set_mesh_parameters(self, **kwargs):
        """Set mesh parameters (no-op for simple grid)."""
        pass

    def export_mesh(self, format_type: str, filename: str):
        """Export mesh in basic format."""
        if not hasattr(self, '_mesh_data') or self._mesh_data is None:
            self._mesh_data = self.generate_mesh()

        if format_type.lower() in ['txt', 'numpy']:
            # Simple text export
            np.savetxt(f"{filename}_vertices.txt", self._mesh_data.vertices)
            np.savetxt(f"{filename}_elements.txt", self._mesh_data.elements, fmt='%d')
        else:
            raise NotImplementedError(f"Export format {format_type} not supported by SimpleGrid2D")

    def generate_mesh(self) -> MeshData:
        """Generate regular 2D grid mesh."""
        # Create coordinate arrays
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)

        # Create mesh grid
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Flatten to get vertices
        vertices = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangular elements (2 triangles per grid cell)
        elements = []
        for i in range(self.nx):
            for j in range(self.ny):
                # Grid indices
                bottom_left = i * (self.ny + 1) + j
                bottom_right = i * (self.ny + 1) + j + 1
                top_left = (i + 1) * (self.ny + 1) + j
                top_right = (i + 1) * (self.ny + 1) + j + 1

                # Two triangles per cell
                elements.append([bottom_left, bottom_right, top_left])
                elements.append([bottom_right, top_right, top_left])

        elements = np.array(elements)

        # Boundary faces (edges for 2D)
        boundary_faces = []
        boundary_tags = []

        # Bottom boundary (y = ymin)
        for i in range(self.nx):
            j = 0
            bottom_left = i * (self.ny + 1) + j
            bottom_right = (i + 1) * (self.ny + 1) + j
            boundary_faces.append([bottom_left, bottom_right])
            boundary_tags.append(1)  # Bottom = 1

        # Right boundary (x = xmax)
        for j in range(self.ny):
            i = self.nx
            bottom_right = i * (self.ny + 1) + j
            top_right = i * (self.ny + 1) + j + 1
            boundary_faces.append([bottom_right, top_right])
            boundary_tags.append(2)  # Right = 2

        # Top boundary (y = ymax)
        for i in range(self.nx):
            j = self.ny
            top_left = i * (self.ny + 1) + j
            top_right = (i + 1) * (self.ny + 1) + j
            boundary_faces.append([top_left, top_right])
            boundary_tags.append(3)  # Top = 3

        # Left boundary (x = xmin)
        for j in range(self.ny):
            i = 0
            bottom_left = i * (self.ny + 1) + j
            top_left = i * (self.ny + 1) + j + 1
            boundary_faces.append([bottom_left, top_left])
            boundary_tags.append(4)  # Left = 4

        boundary_faces = np.array(boundary_faces)
        boundary_tags = np.array(boundary_tags)

        # Element tags (all interior)
        element_tags = np.ones(len(elements), dtype=int)

        # Create mesh data
        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=boundary_tags,
            element_tags=element_tags,
            boundary_faces=boundary_faces,
            dimension=2
        )

        # Compute quality metrics
        mesh_data.element_volumes = self._compute_triangle_areas(vertices, elements)
        mesh_data.quality_metrics = self._compute_quality_metrics(mesh_data)

        return mesh_data

    def _compute_triangle_areas(self, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Compute areas of triangular elements."""
        areas = np.zeros(len(elements))

        for i, elem in enumerate(elements):
            coords = vertices[elem]
            # Area using cross product
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            area = 0.5 * abs(np.cross(v1, v2))
            areas[i] = area

        return areas

    def _compute_quality_metrics(self, mesh_data: MeshData) -> dict:
        """Compute basic quality metrics."""
        areas = mesh_data.element_volumes

        return {
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "mean_area": float(np.mean(areas)),
            "area_ratio": float(np.max(areas) / np.min(areas)) if np.min(areas) > 0 else np.inf,
            "num_elements": len(mesh_data.elements),
            "num_vertices": len(mesh_data.vertices)
        }


class SimpleGrid3D(BaseGeometry):
    """Simple 3D rectangular grid without external mesh dependencies."""

    def __init__(self,
                 bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
                 resolution: tuple[int, int, int] = (16, 16, 16)):
        """
        Initialize 3D grid.

        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            resolution: (nx, ny, nz) grid points
        """
        super().__init__(dimension=3)

        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        self.nx, self.ny, self.nz = resolution
        self._bounds = bounds

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Get domain bounds."""
        return self._bounds

    def create_gmsh_geometry(self):
        """Not implemented for simple grid - raises NotImplementedError."""
        raise NotImplementedError("SimpleGrid3D does not use Gmsh")

    def set_mesh_parameters(self, **kwargs):
        """Set mesh parameters (no-op for simple grid)."""
        pass

    def export_mesh(self, format_type: str, filename: str):
        """Export mesh in basic format."""
        if not hasattr(self, '_mesh_data') or self._mesh_data is None:
            self._mesh_data = self.generate_mesh()

        if format_type.lower() in ['txt', 'numpy']:
            # Simple text export
            np.savetxt(f"{filename}_vertices.txt", self._mesh_data.vertices)
            np.savetxt(f"{filename}_elements.txt", self._mesh_data.elements, fmt='%d')
        else:
            raise NotImplementedError(f"Export format {format_type} not supported by SimpleGrid3D")

    def generate_mesh(self) -> MeshData:
        """Generate regular 3D grid mesh."""
        # Create coordinate arrays
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        z = np.linspace(self.zmin, self.zmax, self.nz + 1)

        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Flatten to get vertices
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Create tetrahedral elements (6 tetrahedra per grid cell)
        elements = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Grid cell vertices
                    v000 = i * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k
                    v001 = i * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k + 1
                    v010 = i * (self.ny + 1) * (self.nz + 1) + (j + 1) * (self.nz + 1) + k
                    v011 = i * (self.ny + 1) * (self.nz + 1) + (j + 1) * (self.nz + 1) + k + 1
                    v100 = (i + 1) * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k
                    v101 = (i + 1) * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k + 1
                    v110 = (i + 1) * (self.ny + 1) * (self.nz + 1) + (j + 1) * (self.nz + 1) + k
                    v111 = (i + 1) * (self.ny + 1) * (self.nz + 1) + (j + 1) * (self.nz + 1) + k + 1

                    # 6 tetrahedra per cube (standard decomposition)
                    tetrahedra = [
                        [v000, v001, v011, v111],
                        [v000, v011, v010, v111],
                        [v000, v010, v110, v111],
                        [v000, v110, v100, v111],
                        [v000, v100, v101, v111],
                        [v000, v101, v001, v111]
                    ]

                    elements.extend(tetrahedra)

        elements = np.array(elements)

        # Simplified boundary faces (just mark boundary vertices)
        boundary_faces = np.array([]).reshape(0, 3)  # Empty for simplicity
        boundary_tags = np.array([])

        # Element tags (all interior)
        element_tags = np.ones(len(elements), dtype=int)

        # Create mesh data
        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_tags=boundary_tags,
            element_tags=element_tags,
            boundary_faces=boundary_faces,
            dimension=3
        )

        # Compute quality metrics
        mesh_data.element_volumes = self._compute_tetrahedron_volumes(vertices, elements)
        mesh_data.quality_metrics = self._compute_quality_metrics(mesh_data)

        return mesh_data

    def _compute_tetrahedron_volumes(self, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Compute volumes of tetrahedral elements."""
        volumes = np.zeros(len(elements))

        for i, elem in enumerate(elements):
            coords = vertices[elem]
            # Volume using scalar triple product: |det(b-a, c-a, d-a)| / 6
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]

            volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            volumes[i] = volume

        return volumes

    def _compute_quality_metrics(self, mesh_data: MeshData) -> dict:
        """Compute basic quality metrics."""
        volumes = mesh_data.element_volumes

        return {
            "min_volume": float(np.min(volumes)),
            "max_volume": float(np.max(volumes)),
            "mean_volume": float(np.mean(volumes)),
            "volume_ratio": float(np.max(volumes) / np.min(volumes)) if np.min(volumes) > 0 else np.inf,
            "num_elements": len(mesh_data.elements),
            "num_vertices": len(mesh_data.vertices)
        }