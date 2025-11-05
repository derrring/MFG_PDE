"""
Tetrahedral Adaptive Mesh Refinement for 3D MFG Problems

This module implements adaptive mesh refinement for tetrahedral meshes in 3D domains,
enabling efficient solution of MFG problems with localized features or singularities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .base_geometry import MeshData
from .geometry_protocol import GeometryType


@dataclass
class TetrahedronElement:
    """
    Tetrahedral element for 3D adaptive mesh refinement.

    Attributes:
        vertices: Indices of the 4 vertices
        center: Centroid coordinates
        volume: Element volume
        quality: Quality metric (radius ratio)
        level: Refinement level (0 = coarsest)
        parent: Parent element index (None for initial mesh)
        children: List of children element indices (empty if not refined)
        marked_for_refinement: Flag for refinement
        marked_for_coarsening: Flag for coarsening
        error_estimate: Local error estimate
    """

    vertices: np.ndarray
    center: np.ndarray
    volume: float
    quality: float
    level: int = 0
    parent: int | None = None
    children: list[int] | None = None
    marked_for_refinement: bool = False
    marked_for_coarsening: bool = False
    error_estimate: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TetrahedralErrorEstimator:
    """
    Error estimation for tetrahedral meshes using gradient recovery techniques.
    """

    def __init__(self, mesh_data: MeshData):
        self.mesh_data = mesh_data
        self.vertices = mesh_data.vertices
        self.elements = mesh_data.elements

    def estimate_error(self, solution: np.ndarray, method: str = "gradient_recovery") -> np.ndarray:
        """
        Estimate local error for each tetrahedral element.

        Args:
            solution: Solution values at mesh vertices
            method: Error estimation method

        Returns:
            Array of error estimates for each element
        """
        if method == "gradient_recovery":
            return self._gradient_recovery_error(solution)
        elif method == "residual_based":
            return self._residual_based_error(solution)
        elif method == "hierarchical":
            return self._hierarchical_error(solution)
        else:
            raise ValueError(f"Unknown error estimation method: {method}")

    def _gradient_recovery_error(self, solution: np.ndarray) -> np.ndarray:
        """Gradient recovery-based error estimation."""
        num_elements = len(self.elements)
        error_estimates = np.zeros(num_elements)

        # Compute element-wise gradients
        element_gradients = self._compute_element_gradients(solution)

        # Recover smoothed gradients at vertices
        recovered_gradients = self._recover_gradients_at_vertices(solution, element_gradients)

        # Compute error as difference between recovered and element gradients
        for elem_idx, element in enumerate(self.elements):
            # Get recovered gradients at element vertices
            vertex_gradients = recovered_gradients[element]
            avg_recovered_gradient = np.mean(vertex_gradients, axis=0)

            # Compare with element gradient
            element_gradient = element_gradients[elem_idx]
            gradient_diff = np.linalg.norm(avg_recovered_gradient - element_gradient)

            # Scale by element size
            element_size = self._compute_element_size(elem_idx)
            error_estimates[elem_idx] = gradient_diff * element_size

        return error_estimates

    def _compute_element_gradients(self, solution: np.ndarray) -> np.ndarray:
        """Compute gradients for each tetrahedral element."""
        num_elements = len(self.elements)
        gradients = np.zeros((num_elements, 3))

        for elem_idx, element in enumerate(self.elements):
            # Get element vertices and solution values
            elem_vertices = self.vertices[element]
            elem_solution = solution[element]

            # Compute gradient using least squares
            # ∇u ≈ (B^T B)^(-1) B^T (u - u₀)
            center = np.mean(elem_vertices, axis=0)
            B = elem_vertices - center  # 4x3 matrix
            u_rel = elem_solution - elem_solution[0]  # Relative to first vertex

            try:
                # Solve least squares problem
                gradient = np.linalg.lstsq(B[1:], u_rel[1:], rcond=None)[0]
                gradients[elem_idx] = gradient
            except np.linalg.LinAlgError:
                gradients[elem_idx] = 0.0

        return gradients

    def _recover_gradients_at_vertices(self, solution: np.ndarray, element_gradients: np.ndarray) -> np.ndarray:
        """Recover smoothed gradients at vertices using weighted averaging."""
        num_vertices = len(self.vertices)
        recovered_gradients = np.zeros((num_vertices, 3))
        vertex_weights = np.zeros(num_vertices)

        for elem_idx, element in enumerate(self.elements):
            element_volume = self._compute_element_volume(elem_idx)
            element_gradient = element_gradients[elem_idx]

            # Add contribution to each vertex of the element
            for vertex_idx in element:
                recovered_gradients[vertex_idx] += element_volume * element_gradient
                vertex_weights[vertex_idx] += element_volume

        # Normalize by weights
        for vertex_idx in range(num_vertices):
            if vertex_weights[vertex_idx] > 1e-12:
                recovered_gradients[vertex_idx] /= vertex_weights[vertex_idx]

        return recovered_gradients

    def _residual_based_error(self, solution: np.ndarray) -> np.ndarray:
        """Residual-based error estimation."""
        # Simplified residual computation
        num_elements = len(self.elements)
        return np.ones(num_elements) * 0.1  # Placeholder

    def _hierarchical_error(self, solution: np.ndarray) -> np.ndarray:
        """Hierarchical basis error estimation."""
        # Placeholder for hierarchical error estimation
        num_elements = len(self.elements)
        return np.ones(num_elements) * 0.1

    def _compute_element_volume(self, elem_idx: int) -> float:
        """Compute volume of tetrahedral element."""
        element = self.elements[elem_idx]
        coords = self.vertices[element]

        # Volume using scalar triple product: |det(b-a, c-a, d-a)| / 6
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        v3 = coords[3] - coords[0]

        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    def _compute_element_size(self, elem_idx: int) -> float:
        """Compute characteristic size of tetrahedral element."""
        return self._compute_element_volume(elem_idx) ** (1 / 3)


class TetrahedralAMRMesh:
    """
    Adaptive mesh refinement for tetrahedral meshes in 3D.
    """

    def __init__(self, initial_mesh: MeshData, max_refinement_level: int = 5):
        """
        Initialize tetrahedral AMR mesh.

        Args:
            initial_mesh: Initial tetrahedral mesh
            max_refinement_level: Maximum allowed refinement level
        """
        self.initial_mesh = initial_mesh
        self.max_refinement_level = max_refinement_level

        # Current mesh state
        self.vertices = initial_mesh.vertices.copy()
        self.elements_list: list[TetrahedronElement] = []
        self.active_elements: list[int] = []  # Indices of non-refined elements

        # AMR bookkeeping
        self.vertex_count = len(self.vertices)
        self.element_count = 0
        self.refinement_history: list[dict] = []

        # Initialize elements
        self._initialize_elements()

        # Error estimator
        self.error_estimator = TetrahedralErrorEstimator(self.current_mesh_data())

    def _initialize_elements(self):
        """Initialize element list from initial mesh."""
        for elem_vertices in self.initial_mesh.elements:
            coords = self.vertices[elem_vertices]
            center = np.mean(coords, axis=0)
            volume = self._compute_tetrahedron_volume(coords)
            quality = self._compute_tetrahedron_quality(coords)

            element = TetrahedronElement(vertices=elem_vertices, center=center, volume=volume, quality=quality, level=0)

            self.elements_list.append(element)
            self.active_elements.append(self.element_count)
            self.element_count += 1

    # GeometryProtocol implementation
    @property
    def dimension(self) -> int:
        """Spatial dimension (3D for tetrahedral mesh)."""
        return 3

    @property
    def geometry_type(self) -> GeometryType:
        """Geometry type (CARTESIAN_GRID for structured adaptive mesh)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (active elements)."""
        return len(self.active_elements)

    def get_spatial_grid(self) -> NDArray:
        """
        Get spatial grid representation.

        Returns:
            numpy array of active element centers (N, 3)
        """
        centers = np.array([self.elements_list[i].center for i in self.active_elements])
        return centers

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method provides AMR-specific configuration for MFGProblem.
        Tetrahedral AMR meshes have dynamic grids with variable element sizes.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of active elements
                - spatial_shape: (num_active_elements,) - flattened for AMR
                - spatial_bounds: None - AMR has dynamic bounds
                - spatial_discretization: None - AMR has variable element sizes
                - legacy_1d_attrs: None - AMR doesn't support legacy attrs

        Added in v0.10.1 for polymorphic geometry handling.
        """
        return {
            "num_spatial_points": len(self.active_elements),
            "spatial_shape": (len(self.active_elements),),
            "spatial_bounds": None,  # AMR has dynamic bounds
            "spatial_discretization": None,  # AMR has variable element sizes
            "legacy_1d_attrs": None,  # AMR doesn't support legacy 1D attributes
        }

    def refine_mesh(
        self, solution: np.ndarray, refinement_fraction: float = 0.3, error_method: str = "gradient_recovery"
    ) -> bool:
        """
        Perform one step of adaptive mesh refinement.

        Args:
            solution: Current solution for error estimation
            refinement_fraction: Fraction of elements to refine
            error_method: Error estimation method

        Returns:
            True if mesh was modified, False otherwise
        """
        # Update error estimator with current mesh
        self.error_estimator = TetrahedralErrorEstimator(self.current_mesh_data())

        # Estimate errors
        active_solution = self._get_active_solution(solution)
        error_estimates = self.error_estimator.estimate_error(active_solution, error_method)

        # Mark elements for refinement
        num_to_refine = max(1, int(len(self.active_elements) * refinement_fraction))
        refinement_threshold = np.partition(error_estimates, -num_to_refine)[-num_to_refine]

        elements_refined = 0
        for i, elem_idx in enumerate(self.active_elements):
            element = self.elements_list[elem_idx]

            if error_estimates[i] >= refinement_threshold and element.level < self.max_refinement_level:
                element.marked_for_refinement = True
                elements_refined += 1

        if elements_refined == 0:
            return False

        # Perform refinement
        self._execute_refinement()

        # Record refinement history
        self.refinement_history.append(
            {
                "elements_refined": elements_refined,
                "total_active_elements": len(self.active_elements),
                "max_error": np.max(error_estimates),
                "mean_error": np.mean(error_estimates),
            }
        )

        return True

    def _execute_refinement(self):
        """Execute marked refinements using octasection."""
        new_elements = []
        elements_to_remove = []

        for elem_idx in self.active_elements:
            element = self.elements_list[elem_idx]

            if element.marked_for_refinement:
                # Create 8 sub-tetrahedra using octasection
                children_indices = self._refine_tetrahedron(elem_idx)
                element.children = children_indices
                element.marked_for_refinement = False
                elements_to_remove.append(elem_idx)
                new_elements.extend(children_indices)

        # Update active elements list
        for elem_idx in elements_to_remove:
            self.active_elements.remove(elem_idx)

        self.active_elements.extend(new_elements)

    def _refine_tetrahedron(self, elem_idx: int) -> list[int]:
        """
        Refine a tetrahedron using octasection (8 sub-tetrahedra).

        Args:
            elem_idx: Index of element to refine

        Returns:
            List of indices of child elements
        """
        parent_element = self.elements_list[elem_idx]
        parent_vertices = parent_element.vertices
        self.vertices[parent_vertices]

        # Create new vertices at edge midpoints and face centers
        edge_midpoints = {}
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        # Add edge midpoints
        for edge in edges:
            v1, v2 = parent_vertices[edge[0]], parent_vertices[edge[1]]
            midpoint = 0.5 * (self.vertices[v1] + self.vertices[v2])

            # Add new vertex
            self.vertices = np.vstack([self.vertices, midpoint])
            edge_midpoints[edge] = self.vertex_count
            self.vertex_count += 1

        # Create 8 sub-tetrahedra (simplified octasection)
        # This is a simplified version - full octasection is more complex
        children_indices = []

        # Create sub-tetrahedra using various combinations of original and new vertices
        sub_tetrahedra = [
            # Corner tetrahedra
            [parent_vertices[0], edge_midpoints[(0, 1)], edge_midpoints[(0, 2)], edge_midpoints[(0, 3)]],
            [parent_vertices[1], edge_midpoints[(0, 1)], edge_midpoints[(1, 2)], edge_midpoints[(1, 3)]],
            [parent_vertices[2], edge_midpoints[(0, 2)], edge_midpoints[(1, 2)], edge_midpoints[(2, 3)]],
            [parent_vertices[3], edge_midpoints[(0, 3)], edge_midpoints[(1, 3)], edge_midpoints[(2, 3)]],
            # Central tetrahedra (simplified)
            [edge_midpoints[(0, 1)], edge_midpoints[(0, 2)], edge_midpoints[(1, 2)], edge_midpoints[(0, 3)]],
            [edge_midpoints[(0, 1)], edge_midpoints[(1, 2)], edge_midpoints[(1, 3)], edge_midpoints[(0, 3)]],
            [edge_midpoints[(0, 2)], edge_midpoints[(1, 2)], edge_midpoints[(2, 3)], edge_midpoints[(0, 3)]],
            [edge_midpoints[(1, 2)], edge_midpoints[(1, 3)], edge_midpoints[(2, 3)], edge_midpoints[(0, 3)]],
        ]

        for sub_tet_vertices in sub_tetrahedra:
            coords = self.vertices[sub_tet_vertices]
            center = np.mean(coords, axis=0)
            volume = self._compute_tetrahedron_volume(coords)
            quality = self._compute_tetrahedron_quality(coords)

            child_element = TetrahedronElement(
                vertices=np.array(sub_tet_vertices),
                center=center,
                volume=volume,
                quality=quality,
                level=parent_element.level + 1,
                parent=elem_idx,
            )

            self.elements_list.append(child_element)
            children_indices.append(self.element_count)
            self.element_count += 1

        return children_indices

    def _get_active_solution(self, solution: np.ndarray) -> np.ndarray:
        """Extract solution values for active elements only."""
        if len(solution) == len(self.vertices):
            return solution
        else:
            # If solution is per-element, extract active elements
            return solution[self.active_elements]

    def current_mesh_data(self) -> MeshData:
        """Get current mesh as MeshData object."""
        # Extract active elements
        active_elements_list = []
        for elem_idx in self.active_elements:
            element = self.elements_list[elem_idx]
            active_elements_list.append(element.vertices)

        active_elements = np.array(active_elements_list, dtype=np.int32)

        # Create boundary information (simplified)
        boundary_faces = np.array([], dtype=np.int32).reshape(0, 3)
        boundary_tags = np.array([], dtype=np.int32)
        element_tags = np.ones(len(active_elements), dtype=int)

        mesh_data = MeshData(
            vertices=self.vertices,
            elements=cast("NDArray[np.integer]", active_elements),
            element_type="tetrahedron",
            boundary_tags=cast("NDArray[np.integer]", boundary_tags),
            element_tags=element_tags,
            boundary_faces=cast("NDArray[np.integer]", boundary_faces),
            dimension=3,
        )

        # Compute quality metrics
        volumes = np.array([self.elements_list[idx].volume for idx in self.active_elements])
        qualities = np.array([self.elements_list[idx].quality for idx in self.active_elements])

        mesh_data.element_volumes = volumes
        mesh_data.quality_metrics = {
            "min_volume": float(np.min(volumes)),
            "max_volume": float(np.max(volumes)),
            "mean_volume": float(np.mean(volumes)),
            "min_quality": float(np.min(qualities)),
            "mean_quality": float(np.mean(qualities)),
            "num_active_elements": len(self.active_elements),
            "num_total_elements": self.element_count,
            "refinement_levels": [self.elements_list[idx].level for idx in self.active_elements],
        }

        return mesh_data

    def _compute_tetrahedron_volume(self, coords: np.ndarray) -> float:
        """Compute volume of tetrahedron."""
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        v3 = coords[3] - coords[0]
        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    def _compute_tetrahedron_quality(self, coords: np.ndarray) -> float:
        """Compute quality metric (radius ratio) for tetrahedron."""
        # Simplified quality metric
        edges = [
            np.linalg.norm(coords[1] - coords[0]),
            np.linalg.norm(coords[2] - coords[0]),
            np.linalg.norm(coords[3] - coords[0]),
            np.linalg.norm(coords[2] - coords[1]),
            np.linalg.norm(coords[3] - coords[1]),
            np.linalg.norm(coords[3] - coords[2]),
        ]

        volume = self._compute_tetrahedron_volume(coords)
        max_edge = max(edges)  # type: ignore[type-var]

        if max_edge > 0:
            return float(volume / (max_edge**3))
        else:
            return 0.0

    def get_refinement_statistics(self) -> dict:
        """Get comprehensive refinement statistics."""
        level_counts: dict[int, int] = {}
        for elem_idx in self.active_elements:
            level = self.elements_list[elem_idx].level
            level_counts[level] = level_counts.get(level, 0) + 1

        qualities = [self.elements_list[idx].quality for idx in self.active_elements]
        volumes = [self.elements_list[idx].volume for idx in self.active_elements]

        return {
            "total_vertices": self.vertex_count,
            "active_elements": len(self.active_elements),
            "total_elements_created": self.element_count,
            "refinement_levels": level_counts,
            "max_level": max(level_counts.keys()) if level_counts else 0,
            "mean_quality": np.mean(qualities),
            "min_quality": np.min(qualities),
            "mean_volume": np.mean(volumes),
            "min_volume": np.min(volumes),
            "refinement_history": self.refinement_history,
        }

    def compute_quality_metrics(self) -> dict[str, float]:
        """Alias for get_refinement_statistics for compatibility."""
        return self.get_refinement_statistics()


def create_tetrahedral_amr_mesh(initial_mesh: MeshData, **kwargs) -> TetrahedralAMRMesh:
    """
    Factory function to create a tetrahedral AMR mesh.

    Args:
        initial_mesh: Initial tetrahedral mesh
        **kwargs: Additional parameters for TetrahedralAMRMesh

    Returns:
        TetrahedralAMRMesh instance
    """
    return TetrahedralAMRMesh(initial_mesh, **kwargs)
