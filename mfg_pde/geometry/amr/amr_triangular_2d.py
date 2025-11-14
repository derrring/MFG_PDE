#!/usr/bin/env python3
"""
Triangular AMR Integration with Existing MFG_PDE Geometry Infrastructure

This module extends the AMR capabilities to work with triangular meshes,
leveraging the existing MeshData and geometry infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.geometry.meshes.mesh_data import MeshData
from mfg_pde.geometry.protocol import GeometryType

from .amr_quadtree_2d import AMRRefinementCriteria, BaseErrorEstimator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import jit

    from mfg_pde.backends.base_backend import BaseBackend

# Always define JAX_AVAILABLE at module level
try:
    import jax.numpy as jnp  # noqa: F401
    from jax import jit  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import numba  # noqa: F401
    from numba import jit as numba_jit  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class TriangleElement:
    """Triangle element for AMR, integrated with existing MeshData."""

    # Basic properties
    element_id: int
    vertices: np.ndarray  # Shape: (3, 2) - coordinates from MeshData
    vertex_ids: np.ndarray  # Shape: (3,) - indices into MeshData.vertices
    level: int = 0

    # AMR hierarchy
    parent_id: int | None = None
    children_ids: list[int] | None = None

    # Solution and error data
    solution_data: dict[str, Any] | None = None
    error_estimate: float = 0.0

    # Geometric properties (computed)
    area: float = 0.0
    centroid: np.ndarray | None = None
    edge_lengths: np.ndarray | None = None
    aspect_ratio: float = 0.0
    min_angle: float = 0.0
    max_angle: float = 0.0

    def __post_init__(self):
        """Compute geometric properties from vertex coordinates."""
        if self.solution_data is None:
            self.solution_data = {}

        self._compute_geometric_properties()

    def _compute_geometric_properties(self):
        """Compute triangle geometric properties."""
        v0, v1, v2 = self.vertices

        # Edge vectors and lengths
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2

        self.edge_lengths = np.array([np.linalg.norm(e01), np.linalg.norm(e12), np.linalg.norm(e20)])

        # Area using cross product
        cross_product = np.cross(e01, e12)
        self.area = float(0.5 * abs(cross_product))

        # Centroid
        self.centroid = (v0 + v1 + v2) / 3.0

        # Quality metrics
        if self.area > 1e-12:
            perimeter = np.sum(self.edge_lengths)
            self.aspect_ratio = perimeter**2 / (12.0 * self.area)

            # Angles using law of cosines
            a, b, c = self.edge_lengths
            angles = []
            edges = [(b, c, a), (c, a, b), (a, b, c)]

            for e1, e2, e3 in edges:
                cos_angle = (e1**2 + e2**2 - e3**2) / (2 * e1 * e2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angles.append(np.arccos(cos_angle))

            self.min_angle = np.min(angles)
            self.max_angle = np.max(angles)
        else:
            self.aspect_ratio = np.inf
            self.min_angle = 0.0
            self.max_angle = np.pi

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf element."""
        return self.children_ids is None or len(self.children_ids) == 0

    @property
    def diameter(self) -> float:
        """Maximum edge length."""
        if self.edge_lengths is not None:
            return float(np.max(self.edge_lengths))
        return 0.0

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        v0, v1, v2 = self.vertices

        # Compute barycentric coordinates
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-12:
            return False

        a = ((v1[1] - v2[1]) * (point[0] - v2[0]) + (v2[0] - v1[0]) * (point[1] - v2[1])) / denom
        b = ((v2[1] - v0[1]) * (point[0] - v2[0]) + (v0[0] - v2[0]) * (point[1] - v2[1])) / denom
        c = 1 - a - b

        return a >= 0 and b >= 0 and c >= 0


class TriangularAMRMesh:
    """
    Triangular AMR mesh integrated with existing MFG_PDE geometry infrastructure.

    This class extends AMR capabilities to work with triangular meshes generated
    by the existing Gmsh → Meshio → PyVista pipeline.
    """

    def __init__(
        self,
        initial_mesh_data: MeshData,
        refinement_criteria: AMRRefinementCriteria | None = None,
        backend: BaseBackend | None = None,
    ):
        """
        Initialize triangular AMR mesh from existing MeshData.

        Args:
            initial_mesh_data: MeshData from existing geometry pipeline
            refinement_criteria: AMR refinement parameters
            backend: Computational backend
        """
        if initial_mesh_data.element_type != "triangle":
            raise ValueError(f"Expected triangle elements, got {initial_mesh_data.element_type}")

        if initial_mesh_data.dimension != 2:
            raise ValueError(f"Expected 2D mesh, got {initial_mesh_data.dimension}D")

        self.initial_mesh_data = initial_mesh_data
        self.criteria = refinement_criteria or AMRRefinementCriteria()
        self.backend = backend

        # Build triangle elements from MeshData
        self.triangles: dict[int, TriangleElement] = {}
        self.leaf_triangles: list[int] = []
        self._next_element_id = 0

        self._build_initial_triangles()

        # AMR statistics
        self.total_triangles = len(self.triangles)
        self.max_level = 0
        self.refinement_history: list[dict[str, Any]] = []

    def _build_initial_triangles(self):
        """Build triangle elements from MeshData."""
        vertices = self.initial_mesh_data.vertices
        elements = self.initial_mesh_data.elements

        for i, element in enumerate(elements):
            if len(element) != 3:
                raise ValueError(f"Expected triangle with 3 vertices, got {len(element)}")

            # Get vertex coordinates
            triangle_vertices = vertices[element]

            triangle = TriangleElement(element_id=i, vertices=triangle_vertices, vertex_ids=element, level=0)

            self.triangles[i] = triangle
            self.leaf_triangles.append(i)
            self._next_element_id = max(self._next_element_id, i + 1)

    # GeometryProtocol implementation
    @property
    def dimension(self) -> int:
        """Spatial dimension (2D for triangular mesh)."""
        return 2

    @property
    def geometry_type(self) -> GeometryType:
        """Geometry type (CARTESIAN_GRID for structured adaptive mesh)."""
        return GeometryType.CARTESIAN_GRID

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (leaf triangles)."""
        return len(self.leaf_triangles)

    def get_spatial_grid(self) -> np.ndarray:
        """
        Get spatial grid representation.

        Returns:
            numpy array of leaf triangle centroids (N, 2)
        """
        centroids = [self.triangles[i].centroid for i in self.leaf_triangles]
        return np.array(centroids)

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method provides AMR-specific configuration for MFGProblem.
        Triangular AMR meshes have dynamic grids with variable element sizes.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of leaf triangles
                - spatial_shape: (num_leaf_triangles,) - flattened for AMR
                - spatial_bounds: None - AMR has dynamic bounds
                - spatial_discretization: None - AMR has variable element sizes
                - legacy_1d_attrs: None - AMR doesn't support legacy attrs

        Added in v0.10.1 for polymorphic geometry handling.
        """
        return {
            "num_spatial_points": len(self.leaf_triangles),
            "spatial_shape": (len(self.leaf_triangles),),
            "spatial_bounds": None,  # AMR has dynamic bounds
            "spatial_discretization": None,  # AMR has variable element sizes
            "legacy_1d_attrs": None,  # AMR doesn't support legacy 1D attributes
        }

    def refine_triangle(self, triangle_id: int, strategy: str = "red") -> list[int]:
        """
        Refine a triangle using specified strategy.

        Args:
            triangle_id: ID of triangle to refine
            strategy: red (4-way) or "green" (2-way) refinement

        Returns:
            List of new triangle IDs
        """
        if triangle_id not in self.triangles:
            raise ValueError(f"Triangle {triangle_id} not found")

        triangle = self.triangles[triangle_id]

        if not triangle.is_leaf:
            raise ValueError(f"Triangle {triangle_id} is not a leaf")

        if strategy == "red":
            children_ids = self._red_refinement(triangle)
        elif strategy.startswith("green"):
            edge_index = int(strategy[-1]) if len(strategy) > 5 else 0
            children_ids = self._green_refinement(triangle, edge_index)
        else:
            raise ValueError(f"Unknown refinement strategy: {strategy}")

        # Update mesh structure
        triangle.children_ids = children_ids
        self.leaf_triangles.remove(triangle_id)
        self.leaf_triangles.extend(children_ids)

        # Update statistics
        self.total_triangles += len(children_ids) - 1
        self.max_level = max(self.max_level, triangle.level + 1)

        return children_ids

    def _red_refinement(self, parent: TriangleElement) -> list[int]:
        """
        Red refinement: divide triangle into 4 similar triangles.

        This is the standard refinement that maintains triangle quality.
        """
        v0, v1, v2 = parent.vertices

        # Edge midpoints
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)

        # Create 4 children
        child_specs = [
            # Corner triangles
            (np.array([v0, m01, m20]), "corner_0"),
            (np.array([m01, v1, m12]), "corner_1"),
            (np.array([m20, m12, v2]), "corner_2"),
            # Central triangle
            (np.array([m01, m12, m20]), "center"),
        ]

        children_ids = []
        for vertices, child_type in child_specs:
            child_id = self._next_element_id
            self._next_element_id += 1

            # Create child triangle
            child = TriangleElement(
                element_id=child_id,
                vertices=vertices,
                vertex_ids=np.array([-1, -1, -1]),  # Virtual vertices
                level=parent.level + 1,
                parent_id=parent.element_id,
            )
            if child.solution_data is not None:
                child.solution_data["refinement_type"] = f"red_{child_type}"

            self.triangles[child_id] = child
            children_ids.append(child_id)

        return children_ids

    def _green_refinement(self, parent: TriangleElement, edge_index: int) -> list[int]:
        """
        Green refinement: divide triangle into 2 triangles by bisecting one edge.

        Used for mesh conformity and anisotropic refinement.
        """
        v0, v1, v2 = parent.vertices
        vertices = [v0, v1, v2]

        # Bisect specified edge
        if edge_index == 0:  # Edge v0-v1
            midpoint = 0.5 * (v0 + v1)
            child1_vertices = np.array([v0, midpoint, v2])
            child2_vertices = np.array([midpoint, v1, v2])
        elif edge_index == 1:  # Edge v1-v2
            midpoint = 0.5 * (v1 + v2)
            child1_vertices = np.array([v0, v1, midpoint])
            child2_vertices = np.array([v0, midpoint, v2])
        else:  # Edge v2-v0
            midpoint = 0.5 * (v2 + v0)
            child1_vertices = np.array([v0, v1, midpoint])
            child2_vertices = np.array([midpoint, v1, v2])

        children_ids = []
        for i, vertices_arr in enumerate([child1_vertices, child2_vertices]):
            vertices = vertices_arr  # type: ignore[assignment]
            child_id = self._next_element_id
            self._next_element_id += 1

            child = TriangleElement(
                element_id=child_id,
                vertices=vertices,  # type: ignore[arg-type]
                vertex_ids=np.array([-1, -1, -1]),  # Virtual vertices
                level=parent.level + 1,
                parent_id=parent.element_id,
            )
            if child.solution_data is not None:
                child.solution_data["refinement_type"] = f"green_{edge_index}_{i}"

            self.triangles[child_id] = child
            children_ids.append(child_id)

        return children_ids

    def adapt_mesh(self, solution_data: dict[str, np.ndarray], error_estimator: BaseErrorEstimator) -> dict[str, int]:
        """
        Adapt triangular mesh based on error estimates.

        Args:
            solution_data: Solution fields (U, M) on triangle centroids
            error_estimator: Error estimation algorithm

        Returns:
            Adaptation statistics
        """
        refinements = 0
        red_refinements = 0
        green_refinements = 0

        # Collect leaf triangles that need refinement
        triangles_to_refine = []

        for triangle_id in self.leaf_triangles.copy():  # Copy since we'll modify during iteration
            triangle = self.triangles[triangle_id]

            # Check refinement constraints
            if triangle.level >= self.criteria.max_refinement_levels:
                continue
            if triangle.diameter < self.criteria.min_cell_size:
                continue

            # Estimate error for this triangle
            error = self._estimate_triangle_error(triangle, solution_data, error_estimator)
            triangle.error_estimate = error

            # Refinement decision
            if error > self.criteria.error_threshold:
                triangles_to_refine.append((triangle_id, triangle))

        # Execute refinements
        for triangle_id, triangle in triangles_to_refine:
            # Choose refinement strategy based on triangle quality
            if triangle.aspect_ratio < 3.0 and triangle.min_angle > np.pi / 9:  # ~20 degrees
                # Good quality triangle - use red refinement
                self.refine_triangle(triangle_id, "red")
                red_refinements += 1
            else:
                # Poor quality triangle - use green refinement on longest edge
                if triangle.edge_lengths is not None:
                    longest_edge = np.argmax(triangle.edge_lengths)
                else:
                    longest_edge = np.int64(0)
                self.refine_triangle(triangle_id, f"green{longest_edge}")
                green_refinements += 1

            refinements += 1

        # Record adaptation statistics
        stats = {
            "total_refined": refinements,
            "red_refinements": red_refinements,
            "green_refinements": green_refinements,
            "total_coarsened": 0,  # Not implemented yet
            "final_triangles": self.total_triangles,
            "max_level": self.max_level,
        }

        self.refinement_history.append(stats)
        return stats

    def _estimate_triangle_error(
        self,
        triangle: TriangleElement,
        solution_data: dict[str, np.ndarray],
        error_estimator: BaseErrorEstimator,
    ) -> float:
        """
        Estimate error for a single triangle.

        This integrates with the existing error estimation framework.
        """

        # Create pseudo-node for compatibility with existing error estimator
        class PseudoNode:
            def __init__(self, triangle: TriangleElement):
                self.dx = triangle.diameter
                self.dy = triangle.diameter
                self.area = triangle.area
                if triangle.centroid is not None:
                    self.center_x = triangle.centroid[0]
                    self.center_y = triangle.centroid[1]
                else:
                    self.center_x = 0.0
                    self.center_y = 0.0

        pseudo_node = PseudoNode(triangle)

        # Use existing error estimator - cast to expected type

        return error_estimator.estimate_error(pseudo_node, solution_data)  # type: ignore[arg-type]

    def get_mesh_statistics(self) -> dict[str, Any]:
        """Get comprehensive mesh statistics."""
        # Level distribution
        level_counts: dict[int, int] = {}
        total_area = 0.0
        min_aspect_ratio = np.inf
        max_aspect_ratio = 0.0

        for triangle in self.triangles.values():
            if triangle.is_leaf:
                level = triangle.level
                level_counts[level] = level_counts.get(level, 0) + 1
                total_area += triangle.area

                if triangle.aspect_ratio < min_aspect_ratio:
                    min_aspect_ratio = triangle.aspect_ratio
                if triangle.aspect_ratio > max_aspect_ratio:
                    max_aspect_ratio = triangle.aspect_ratio

        return {
            "total_triangles": self.total_triangles,
            "leaf_triangles": len(self.leaf_triangles),
            "max_level": self.max_level,
            "level_distribution": level_counts,
            "total_area": total_area,
            "min_aspect_ratio": min_aspect_ratio,
            "max_aspect_ratio": max_aspect_ratio,
            "refinement_ratio": (max_aspect_ratio / min_aspect_ratio if min_aspect_ratio > 0 else 1.0),
            "initial_elements": len(self.initial_mesh_data.elements),
            "refinement_history": self.refinement_history,
        }

    def export_to_mesh_data(self) -> MeshData:
        """
        Export adapted triangular mesh back to MeshData format.

        This allows integration with existing visualization and analysis tools.
        """
        # Collect all leaf triangles
        leaf_triangles = [self.triangles[tid] for tid in self.leaf_triangles]

        # Build vertices and elements arrays
        vertices = []
        elements = []
        vertex_map = {}  # Map (x, y) -> vertex_index
        next_vertex_id = 0

        for triangle in leaf_triangles:
            element = []
            for vertex in triangle.vertices:
                vertex_key = (vertex[0], vertex[1])

                if vertex_key not in vertex_map:
                    vertex_map[vertex_key] = next_vertex_id
                    vertices.append(vertex)
                    next_vertex_id += 1

                element.append(vertex_map[vertex_key])

            elements.append(element)

        # Create new MeshData
        return MeshData(
            vertices=np.array(vertices),
            elements=np.array(elements),
            element_type="triangle",
            boundary_tags=np.array([]),  # Would need boundary identification
            element_tags=np.array(list(self.leaf_triangles)),
            boundary_faces=np.array([]),  # Would need boundary face extraction
            dimension=2,
            metadata={
                "amr_adapted": True,
                "max_level": self.max_level,
                "total_refinements": len(self.refinement_history),
                "original_elements": len(self.initial_mesh_data.elements),
            },
        )


# Integration with existing error estimators
class TriangularMeshErrorEstimator(BaseErrorEstimator):
    """Error estimator specifically designed for triangular meshes."""

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend
        self.use_jax = JAX_AVAILABLE and (backend is None or getattr(backend, "name", "") == "jax")

    def estimate_error(self, node, solution_data: dict[str, np.ndarray]) -> float:
        """
        Estimate error for triangular element.

        Uses gradient-based error estimation adapted for triangular meshes.
        """
        if "U" not in solution_data or "M" not in solution_data:
            return 0.0

        # For triangular meshes, we need different approach than structured grids
        # This is a simplified implementation - would need proper FEM gradient recovery

        U = solution_data["U"]
        M = solution_data["M"]

        # Sample solution at element center
        # In practice, would interpolate from surrounding elements
        getattr(node, "center_x", 0.0)
        getattr(node, "center_y", 0.0)

        # Estimate local gradients (simplified)
        h = getattr(node, "dx", 1.0)

        # Use finite difference approximation
        if U.ndim == 2 and U.size > 1:
            i = min(int(U.shape[0] / 2), U.shape[0] - 1)
            j = min(int(U.shape[1] / 2), U.shape[1] - 1)

            u_val = U[i, j]
            m_val = M[i, j]

            # Gradient magnitude estimate
            grad_estimate = abs(u_val) + abs(m_val)
        else:
            grad_estimate = 1.0

        # Scale by element size
        return h * grad_estimate


# Factory function for triangular AMR
def create_triangular_amr_mesh(
    mesh_data: MeshData,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    backend: BaseBackend | None = None,
) -> TriangularAMRMesh:
    """
    Create triangular AMR mesh from existing MeshData.

    Args:
        mesh_data: Triangular mesh from existing geometry pipeline
        error_threshold: Error threshold for refinement
        max_levels: Maximum refinement levels
        backend: Computational backend

    Returns:
        TriangularAMRMesh ready for adaptive refinement
    """
    criteria = AMRRefinementCriteria(error_threshold=error_threshold, max_refinement_levels=max_levels)

    return TriangularAMRMesh(initial_mesh_data=mesh_data, refinement_criteria=criteria, backend=backend)
