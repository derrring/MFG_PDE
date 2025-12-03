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
from mfg_pde.geometry.meshes.mesh_data import MeshData
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
            resolution: (nx, ny) number of intervals (grid points = nx+1, ny+1)
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

    def get_multi_index(self, flat_index: int) -> tuple[int, int]:
        """
        Convert flat index to 2D grid indices (i, j).

        Assumes C-order (row-major) indexing where the grid is shaped (nx+1, ny+1).

        Args:
            flat_index: Flat index in [0, (nx+1)*(ny+1))

        Returns:
            Tuple (i, j) of grid indices

        Example:
            >>> grid = SimpleGrid2D(bounds=(0, 1, 0, 1), resolution=(10, 10))
            >>> i, j = grid.get_multi_index(53)  # Convert flat index to (i,j)
        """
        total_points = (self.nx + 1) * (self.ny + 1)
        if flat_index < 0 or flat_index >= total_points:
            raise ValueError(f"flat_index {flat_index} out of range [0, {total_points})")

        # C-order (row-major): index = i * (ny+1) + j
        i = flat_index // (self.ny + 1)
        j = flat_index % (self.ny + 1)

        return i, j

    def get_index(self, multi_index: tuple[int, ...]) -> int:
        """
        Convert 2D grid indices (i, j) to flat index.

        Inverse of get_multi_index(). Uses C-order (row-major) indexing.

        Args:
            multi_index: Tuple (i, j) of grid indices

        Returns:
            Flat index in [0, (nx+1)*(ny+1))

        Example:
            >>> grid = SimpleGrid2D(bounds=(0, 1, 0, 1), resolution=(10, 10))
            >>> flat_idx = grid.get_index((4, 9))  # Convert (i,j) to flat index
        """
        i, j = multi_index
        if i < 0 or i > self.nx or j < 0 or j > self.ny:
            raise ValueError(f"Grid index ({i}, {j}) out of range")

        # C-order (row-major): index = i * (ny+1) + j
        return i * (self.ny + 1) + j

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


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing SimpleGrid2D...")

    import numpy as np

    # Test grid creation
    grid = SimpleGrid2D(bounds=(0.0, 10.0, 0.0, 5.0), resolution=(11, 6))

    assert grid.dimension == 2
    assert grid.nx == 11
    assert grid.ny == 6
    assert np.isclose(grid.dx, 10.0 / 11)
    assert np.isclose(grid.dy, 5.0 / 6)

    print(f"  Grid: {grid.nx}×{grid.ny}, spacing dx={grid.dx:.3f}, dy={grid.dy:.3f}")

    # Test coordinates
    coords = grid.coordinates
    assert len(coords) == 2
    assert coords[0].shape == (grid.nx + 1,)
    assert coords[1].shape == (grid.ny + 1,)

    print(f"  Coordinates: x has {coords[0].shape[0]} points, y has {coords[1].shape[0]} points")

    # Test bounds
    min_coords, max_coords = grid.bounds
    assert np.allclose(min_coords, [0.0, 0.0])
    assert np.allclose(max_coords, [10.0, 5.0])

    print(f"  Bounds: [{min_coords[0]:.1f}, {max_coords[0]:.1f}] × [{min_coords[1]:.1f}, {max_coords[1]:.1f}]")

    print("Smoke tests passed!")
