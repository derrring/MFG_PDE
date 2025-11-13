"""
Simple grid geometry for high-dimensional MFG problems.

This module provides basic rectangular grid generation without external dependencies,
suitable for regular domain problems and testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

from mfg_pde.geometry.base import CartesianGrid
from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.protocol import GeometryType


class SimpleGrid2D(CartesianGrid):
    """Simple 2D rectangular grid without external mesh dependencies."""

    def __init__(
        self, bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0), resolution: tuple[int, int] = (32, 32)
    ):
        """
        Initialize 2D grid.

        Args:
            bounds: (xmin, xmax, ymin, ymax)
            resolution: (nx, ny) grid points
        """
        self._dimension = 2
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.nx, self.ny = resolution
        self._bounds = bounds
        self._mesh_data: MeshData | None = None

        # Compute grid spacing
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get domain bounds."""
        min_coords = np.array([self.xmin, self.ymin])
        max_coords = np.array([self.xmax, self.ymax])
        return min_coords, max_coords

    @property
    def coordinates(self) -> list[NDArray]:
        """
        Get coordinate arrays for each dimension (TensorProductGrid compatible).

        Returns:
            List of 1D coordinate arrays [x_coords, y_coords]
        """
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        return [x, y]

    def create_gmsh_geometry(self):
        """Not implemented for simple grid - raises NotImplementedError."""
        raise NotImplementedError("SimpleGrid2D does not use Gmsh")

    def set_mesh_parameters(self, **kwargs):
        """Set mesh parameters (no-op for simple grid)."""

    def export_mesh(self, format_type: str, filename: str):
        """Export mesh in basic format."""
        if not hasattr(self, "_mesh_data") or self._mesh_data is None:
            self._mesh_data = self.generate_mesh()

        if format_type.lower() in ["txt", "numpy"]:
            # Simple text export
            np.savetxt(f"{filename}_vertices.txt", self._mesh_data.vertices)
            np.savetxt(f"{filename}_elements.txt", self._mesh_data.elements, fmt="%d")
        else:
            raise NotImplementedError(f"Export format {format_type} not supported by SimpleGrid2D")

    def generate_mesh(self) -> MeshData:
        """Generate regular 2D grid mesh."""
        # Create coordinate arrays
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)

        # Create mesh grid
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Flatten to get vertices
        vertices = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangular elements (2 triangles per grid cell)
        elements_list: list[list[int]] = []
        for i in range(self.nx):
            for j in range(self.ny):
                # Grid indices
                bottom_left = i * (self.ny + 1) + j
                bottom_right = i * (self.ny + 1) + j + 1
                top_left = (i + 1) * (self.ny + 1) + j
                top_right = (i + 1) * (self.ny + 1) + j + 1

                # Two triangles per cell
                elements_list.append([bottom_left, bottom_right, top_left])
                elements_list.append([bottom_right, top_right, top_left])

        elements = np.array(elements_list, dtype=np.int32)

        # Boundary faces (edges for 2D)
        boundary_faces_list: list[list[int]] = []
        boundary_tags_list: list[int] = []

        # Bottom boundary (y = ymin)
        for i in range(self.nx):
            j = 0
            bottom_left = i * (self.ny + 1) + j
            bottom_right = (i + 1) * (self.ny + 1) + j
            boundary_faces_list.append([bottom_left, bottom_right])
            boundary_tags_list.append(1)  # Bottom = 1

        # Right boundary (x = xmax)
        for j in range(self.ny):
            i = self.nx
            bottom_right = i * (self.ny + 1) + j
            top_right = i * (self.ny + 1) + j + 1
            boundary_faces_list.append([bottom_right, top_right])
            boundary_tags_list.append(2)  # Right = 2

        # Top boundary (y = ymax)
        for i in range(self.nx):
            j = self.ny
            top_left = i * (self.ny + 1) + j
            top_right = (i + 1) * (self.ny + 1) + j
            boundary_faces_list.append([top_left, top_right])
            boundary_tags_list.append(3)  # Top = 3

        # Left boundary (x = xmin)
        for j in range(self.ny):
            i = 0
            bottom_left = i * (self.ny + 1) + j
            top_left = i * (self.ny + 1) + j + 1
            boundary_faces_list.append([bottom_left, top_left])
            boundary_tags_list.append(4)  # Left = 4

        boundary_faces = np.array(boundary_faces_list, dtype=np.int32)
        boundary_tags = np.array(boundary_tags_list, dtype=np.int32)

        # Element tags (all interior)
        element_tags = np.ones(len(elements), dtype=int)

        # Create mesh data
        mesh_data = MeshData(
            vertices=vertices,
            elements=cast("NDArray[np.integer]", elements),
            element_type="triangle",
            boundary_tags=cast("NDArray[np.integer]", boundary_tags),
            element_tags=element_tags,
            boundary_faces=cast("NDArray[np.integer]", boundary_faces),
            dimension=2,
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
            "num_vertices": len(mesh_data.vertices),
        }

    # ============================================================================
    # Geometry ABC implementation
    # ============================================================================

    @property
    def dimension(self) -> int:
        """Spatial dimension (always 2 for SimpleGrid2D)."""
        return self._dimension

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (Cartesian grid)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of grid points."""
        return (self.nx + 1) * (self.ny + 1)

    def get_spatial_grid(self) -> NDArray:
        """Get spatial grid as (N, 2) array of vertices."""
        if self._mesh_data is None:
            self._mesh_data = self.generate_mesh()
        return self._mesh_data.vertices

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """Get bounding box."""
        return self.bounds

    def get_problem_config(self) -> dict:
        """Return configuration for MFGProblem."""
        return {
            "num_spatial_points": self.num_spatial_points,
            "spatial_shape": (self.nx + 1, self.ny + 1),
            "spatial_bounds": [(self.xmin, self.xmax), (self.ymin, self.ymax)],
            "spatial_discretization": [self.nx, self.ny],
            "legacy_1d_attrs": None,
        }

    # ============================================================================
    # CartesianGrid ABC implementation
    # ============================================================================

    def get_grid_spacing(self) -> list[float]:
        """Get grid spacing [dx, dy]."""
        return [self.dx, self.dy]

    def get_grid_shape(self) -> tuple[int, ...]:
        """Get grid shape (nx+1, ny+1)."""
        return (self.nx + 1, self.ny + 1)

    # ============================================================================
    # Solver Operation Interface
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """Return finite difference Laplacian for 2D regular grid."""

        def laplacian_2d(u: NDArray, idx: tuple[int, int]) -> float:
            """
            Compute 2D Laplacian using central differences.

            Args:
                u: Solution array of shape (nx+1, ny+1)
                idx: Grid index (i, j)

            Returns:
                Laplacian value
            """
            i, j = idx
            nx1, ny1 = u.shape

            # Boundary clamping
            i_plus = min(i + 1, nx1 - 1)
            i_minus = max(i - 1, 0)
            j_plus = min(j + 1, ny1 - 1)
            j_minus = max(j - 1, 0)

            # Central differences
            laplacian = (u[i_plus, j] - 2.0 * u[i, j] + u[i_minus, j]) / (self.dx**2)
            laplacian += (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (self.dy**2)

            return float(laplacian)

        return laplacian_2d

    def get_gradient_operator(self) -> Callable:
        """Return finite difference gradient for 2D regular grid."""

        def gradient_2d(u: NDArray, idx: tuple[int, int]) -> NDArray:
            """
            Compute 2D gradient using central differences.

            Args:
                u: Solution array of shape (nx+1, ny+1)
                idx: Grid index (i, j)

            Returns:
                Gradient [du/dx, du/dy]
            """
            i, j = idx
            nx1, ny1 = u.shape

            # Boundary clamping
            i_plus = min(i + 1, nx1 - 1)
            i_minus = max(i - 1, 0)
            j_plus = min(j + 1, ny1 - 1)
            j_minus = max(j - 1, 0)

            # Central differences
            du_dx = (u[i_plus, j] - u[i_minus, j]) / (2.0 * self.dx)
            du_dy = (u[i, j_plus] - u[i, j_minus]) / (2.0 * self.dy)

            return np.array([du_dx, du_dy])

        return gradient_2d

    def get_interpolator(self) -> Callable:
        """Return bilinear interpolator for 2D grid."""

        def interpolate_2d(u: NDArray, points: NDArray) -> NDArray | float:
            """
            Bilinear interpolation on 2D grid.

            Args:
                u: Solution array of shape (nx+1, ny+1)
                points: Physical coordinates [x, y] or array of points (N, 2)

            Returns:
                Interpolated value(s) - float if single point, array if multiple
            """
            from scipy.interpolate import RegularGridInterpolator

            x = np.linspace(self.xmin, self.xmax, self.nx + 1)
            y = np.linspace(self.ymin, self.ymax, self.ny + 1)

            interpolator = RegularGridInterpolator((x, y), u, method="linear", bounds_error=False, fill_value=0.0)

            # Handle both single point and multiple points
            result = interpolator(points)

            # Return scalar for single point, array for multiple
            if result.shape == ():
                return float(result)
            return result

        return interpolate_2d

    def get_boundary_handler(self):
        """Return boundary handler (placeholder)."""
        return {"type": "simple_grid_2d", "implementation": "placeholder"}


class SimpleGrid3D(CartesianGrid):
    """Simple 3D rectangular grid without external mesh dependencies."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        resolution: tuple[int, int, int] = (16, 16, 16),
    ):
        """
        Initialize 3D grid.

        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            resolution: (nx, ny, nz) grid points
        """
        self._dimension = 3
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        self.nx, self.ny, self.nz = resolution
        self._bounds = bounds

        # Compute grid spacing
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.dz = (self.zmax - self.zmin) / self.nz
        self._mesh_data: MeshData | None = None

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get domain bounds."""
        min_coords = np.array([self.xmin, self.ymin, self.zmin])
        max_coords = np.array([self.xmax, self.ymax, self.zmax])
        return min_coords, max_coords

    @property
    def coordinates(self) -> list[NDArray]:
        """
        Get coordinate arrays for each dimension (TensorProductGrid compatible).

        Returns:
            List of 1D coordinate arrays [x_coords, y_coords, z_coords]
        """
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        z = np.linspace(self.zmin, self.zmax, self.nz + 1)
        return [x, y, z]

    def create_gmsh_geometry(self):
        """Not implemented for simple grid - raises NotImplementedError."""
        raise NotImplementedError("SimpleGrid3D does not use Gmsh")

    def set_mesh_parameters(self, **kwargs):
        """Set mesh parameters (no-op for simple grid)."""

    def export_mesh(self, format_type: str, filename: str):
        """Export mesh in basic format."""
        if not hasattr(self, "_mesh_data") or self._mesh_data is None:
            self._mesh_data = self.generate_mesh()

        if format_type.lower() in ["txt", "numpy"]:
            # Simple text export
            np.savetxt(f"{filename}_vertices.txt", self._mesh_data.vertices)
            np.savetxt(f"{filename}_elements.txt", self._mesh_data.elements, fmt="%d")
        else:
            raise NotImplementedError(f"Export format {format_type} not supported by SimpleGrid3D")

    def generate_mesh(self) -> MeshData:
        """Generate regular 3D grid mesh."""
        # Create coordinate arrays
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        z = np.linspace(self.zmin, self.zmax, self.nz + 1)

        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Flatten to get vertices
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Create tetrahedral elements (6 tetrahedra per grid cell)
        elements_list: list[list[int]] = []
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
                        [v000, v101, v001, v111],
                    ]

                    elements_list.extend(tetrahedra)

        elements = np.array(elements_list, dtype=np.int32)

        # Simplified boundary faces (just mark boundary vertices)
        boundary_faces = np.array([], dtype=np.int32).reshape(0, 3)  # Empty for simplicity
        boundary_tags = np.array([], dtype=np.int32)

        # Element tags (all interior)
        element_tags = np.ones(len(elements), dtype=int)

        # Create mesh data
        mesh_data = MeshData(
            vertices=vertices,
            elements=cast("NDArray[np.integer]", elements),
            element_type="tetrahedron",
            boundary_tags=cast("NDArray[np.integer]", boundary_tags),
            element_tags=element_tags,
            boundary_faces=cast("NDArray[np.integer]", boundary_faces),
            dimension=3,
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
            "num_vertices": len(mesh_data.vertices),
        }

    # ============================================================================
    # Geometry ABC implementation
    # ============================================================================

    @property
    def dimension(self) -> int:
        """Spatial dimension (always 3 for SimpleGrid3D)."""
        return self._dimension

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (Cartesian grid)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of grid points."""
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def get_spatial_grid(self) -> NDArray:
        """Get spatial grid as (N, 3) array of vertices."""
        if self._mesh_data is None:
            self._mesh_data = self.generate_mesh()
        return self._mesh_data.vertices

    def get_bounds(self) -> tuple[NDArray, NDArray]:
        """Get bounding box."""
        return self.bounds

    def get_problem_config(self) -> dict:
        """Return configuration for MFGProblem."""
        return {
            "num_spatial_points": self.num_spatial_points,
            "spatial_shape": (self.nx + 1, self.ny + 1, self.nz + 1),
            "spatial_bounds": [(self.xmin, self.xmax), (self.ymin, self.ymax), (self.zmin, self.zmax)],
            "spatial_discretization": [self.nx, self.ny, self.nz],
            "legacy_1d_attrs": None,
        }

    # ============================================================================
    # CartesianGrid ABC implementation
    # ============================================================================

    def get_grid_spacing(self) -> list[float]:
        """Get grid spacing [dx, dy, dz]."""
        return [self.dx, self.dy, self.dz]

    def get_grid_shape(self) -> tuple[int, ...]:
        """Get grid shape (nx+1, ny+1, nz+1)."""
        return (self.nx + 1, self.ny + 1, self.nz + 1)

    # ============================================================================
    # Solver Operation Interface
    # ============================================================================

    def get_laplacian_operator(self) -> Callable:
        """Return finite difference Laplacian for 3D regular grid."""

        def laplacian_3d(u: NDArray, idx: tuple[int, int, int]) -> float:
            """
            Compute 3D Laplacian using central differences.

            Args:
                u: Solution array of shape (nx+1, ny+1, nz+1)
                idx: Grid index (i, j, k)

            Returns:
                Laplacian value
            """
            i, j, k = idx
            nx1, ny1, nz1 = u.shape

            # Boundary clamping
            i_plus = min(i + 1, nx1 - 1)
            i_minus = max(i - 1, 0)
            j_plus = min(j + 1, ny1 - 1)
            j_minus = max(j - 1, 0)
            k_plus = min(k + 1, nz1 - 1)
            k_minus = max(k - 1, 0)

            # Central differences in each dimension
            laplacian = (u[i_plus, j, k] - 2.0 * u[i, j, k] + u[i_minus, j, k]) / (self.dx**2)
            laplacian += (u[i, j_plus, k] - 2.0 * u[i, j, k] + u[i, j_minus, k]) / (self.dy**2)
            laplacian += (u[i, j, k_plus] - 2.0 * u[i, j, k] + u[i, j, k_minus]) / (self.dz**2)

            return float(laplacian)

        return laplacian_3d

    def get_gradient_operator(self) -> Callable:
        """Return finite difference gradient for 3D regular grid."""

        def gradient_3d(u: NDArray, idx: tuple[int, int, int]) -> NDArray:
            """
            Compute 3D gradient using central differences.

            Args:
                u: Solution array of shape (nx+1, ny+1, nz+1)
                idx: Grid index (i, j, k)

            Returns:
                Gradient [du/dx, du/dy, du/dz]
            """
            i, j, k = idx
            nx1, ny1, nz1 = u.shape

            # Boundary clamping
            i_plus = min(i + 1, nx1 - 1)
            i_minus = max(i - 1, 0)
            j_plus = min(j + 1, ny1 - 1)
            j_minus = max(j - 1, 0)
            k_plus = min(k + 1, nz1 - 1)
            k_minus = max(k - 1, 0)

            # Central differences
            du_dx = (u[i_plus, j, k] - u[i_minus, j, k]) / (2.0 * self.dx)
            du_dy = (u[i, j_plus, k] - u[i, j_minus, k]) / (2.0 * self.dy)
            du_dz = (u[i, j, k_plus] - u[i, j, k_minus]) / (2.0 * self.dz)

            return np.array([du_dx, du_dy, du_dz])

        return gradient_3d

    def get_interpolator(self) -> Callable:
        """Return trilinear interpolator for 3D grid."""

        def interpolate_3d(u: NDArray, points: NDArray) -> NDArray | float:
            """
            Trilinear interpolation on 3D grid.

            Args:
                u: Solution array of shape (nx+1, ny+1, nz+1)
                points: Physical coordinates [x, y, z] or array of points (N, 3)

            Returns:
                Interpolated value(s) - float if single point, array if multiple
            """
            from scipy.interpolate import RegularGridInterpolator

            x = np.linspace(self.xmin, self.xmax, self.nx + 1)
            y = np.linspace(self.ymin, self.ymax, self.ny + 1)
            z = np.linspace(self.zmin, self.zmax, self.nz + 1)

            interpolator = RegularGridInterpolator((x, y, z), u, method="linear", bounds_error=False, fill_value=0.0)

            # Handle both single point and multiple points
            result = interpolator(points)

            # Return scalar for single point, array for multiple
            if result.shape == ():
                return float(result)
            return result

        return interpolate_3d

    def get_boundary_handler(self):
        """Return boundary handler (placeholder)."""
        return {"type": "simple_grid_3d", "implementation": "placeholder"}
