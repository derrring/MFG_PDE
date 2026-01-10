"""
2D boundary condition handling for Mean Field Games.

This module provides comprehensive boundary condition support for 2D MFG problems,
bridging the gap between 1D and 3D implementations with specialized 2D features.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.geometry.meshes.mesh_data import MeshData


class BoundaryCondition2D(ABC):
    """
    Abstract base class for 2D boundary conditions.

    Provides the interface for implementing various boundary condition types
    including Dirichlet, Neumann, Robin, and periodic conditions in 2D.
    """

    def __init__(self, name: str, region_id: int | None = None):
        """
        Initialize boundary condition.

        Args:
            name: Human-readable name for the boundary condition
            region_id: Optional region identifier for multi-region boundaries
        """
        self.name = name
        self.region_id = region_id
        self._direct_vertices: np.ndarray | None = None

    @abstractmethod
    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply boundary condition to system matrix."""

    @abstractmethod
    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Apply boundary condition to right-hand side vector."""

    @abstractmethod
    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate that boundary condition is compatible with mesh."""


class DirichletBC2D(BoundaryCondition2D):
    """
    Dirichlet boundary condition: u = g(x,y,t) on boundary.

    Enforces fixed values on the boundary, commonly used for
    specifying known solution values at domain boundaries.
    """

    def __init__(self, value_function: float | Callable, name: str = "Dirichlet", region_id: int | None = None):
        """
        Initialize Dirichlet boundary condition.

        Args:
            value_function: Either constant value or function(x,y,t) -> float
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        if callable(value_function):
            self.value_function = value_function
        else:
            self.value_function = lambda x, y, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Dirichlet condition by setting diagonal entries to 1."""
        matrix_mod = matrix.tolil()

        for idx in boundary_indices:
            # Clear row and set diagonal to 1
            matrix_mod[idx, :] = 0
            matrix_mod[idx, idx] = 1

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Apply Dirichlet values to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        for i, idx in enumerate(boundary_indices):
            x, y = boundary_vertices[i, :2]  # Take first 2 coordinates for 2D
            rhs_mod[idx] = self.value_function(x, y, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has boundary markers."""
        # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
        try:
            _ = mesh.boundary_markers
            return True
        except AttributeError:
            pass
        try:
            _ = mesh.vertex_markers
            return True
        except AttributeError:
            pass
        return False


class NeumannBC2D(BoundaryCondition2D):
    """
    Neumann boundary condition: ∇u·n = g(x,y,t) on boundary.

    Enforces flux conditions on the boundary, commonly used for
    no-flux conditions or specified gradient conditions.
    """

    def __init__(self, flux_function: float | Callable, name: str = "Neumann", region_id: int | None = None):
        """
        Initialize Neumann boundary condition.

        Args:
            flux_function: Either constant flux or function(x,y,t) -> float
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        if callable(flux_function):
            self.flux_function = flux_function
        else:
            self.flux_function = lambda x, y, t: float(flux_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Neumann condition using ghost point method."""
        # Neumann conditions typically don't modify the matrix structure
        # Implementation depends on discretization scheme
        return matrix

    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Apply Neumann flux to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        # This is a simplified implementation - would need proper boundary integration
        for i, idx in enumerate(boundary_indices):
            x, y = boundary_vertices[i, :2]
            # Add flux contribution (needs proper scaling by boundary length)
            rhs_mod[idx] += self.flux_function(x, y, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has surface normal information."""
        # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
        try:
            _ = mesh.boundary_normals
            return True
        except AttributeError:
            pass
        try:
            _ = mesh.edge_normals
            return True
        except AttributeError:
            pass
        return False


class RobinBC2D(BoundaryCondition2D):
    """
    Robin boundary condition: α·u + β·∇u·n = g(x,y,t) on boundary.

    Mixed boundary condition combining Dirichlet and Neumann terms,
    commonly used for radiation or convection conditions.
    """

    def __init__(
        self,
        alpha: float | Callable,
        beta: float | Callable,
        value_function: float | Callable,
        name: str = "Robin",
        region_id: int | None = None,
    ):
        """
        Initialize Robin boundary condition.

        Args:
            alpha: Coefficient for u term
            beta: Coefficient for ∇u·n term
            value_function: RHS function g(x,y,t)
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)

        # Convert to functions if constants
        self.alpha = alpha if callable(alpha) else lambda x, y, t: float(alpha)
        self.beta = beta if callable(beta) else lambda x, y, t: float(beta)
        self.value_function = value_function if callable(value_function) else lambda x, y, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Robin condition to matrix."""
        matrix_mod = matrix.tolil()

        boundary_vertices = mesh.vertices[boundary_indices]

        for i, idx in enumerate(boundary_indices):
            x, y = boundary_vertices[i, :2]
            alpha_val = self.alpha(x, y, 0.0)  # Time-independent for matrix
            self.beta(x, y, 0.0)

            # Modify diagonal entry for α·u term
            matrix_mod[idx, idx] += alpha_val

            # Add β·∇u·n contribution (simplified implementation)
            # In practice, would need proper gradient discretization

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Apply Robin condition to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        for i, idx in enumerate(boundary_indices):
            x, y = boundary_vertices[i, :2]
            rhs_mod[idx] += self.value_function(x, y, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh compatibility."""
        return True  # Robin conditions are generally compatible


class PeriodicBC2D(BoundaryCondition2D):
    """
    Periodic boundary condition: u(x1,y) = u(x2,y) for paired boundaries.

    Enforces periodicity across opposite edges of the domain,
    commonly used for modeling infinite or repeating domains.
    """

    def __init__(self, paired_boundaries: list[tuple[int, int]], name: str = "Periodic", region_id: int | None = None):
        """
        Initialize periodic boundary condition.

        Args:
            paired_boundaries: List of (boundary1_id, boundary2_id) pairs
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        self.paired_boundaries = paired_boundaries

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply periodic condition by coupling paired boundaries."""
        matrix_mod = matrix.tolil()

        # Get boundary vertex mapping for periodic pairing
        boundary_mapping = self._identify_boundary_vertices_for_pairing(mesh)

        # Apply periodic coupling for each pair
        for boundary1_id, boundary2_id in self.paired_boundaries:
            # Get vertices for each boundary in the pair
            boundary1_vertices = boundary_mapping.get(boundary1_id, np.array([]))
            boundary2_vertices = boundary_mapping.get(boundary2_id, np.array([]))

            # Find corresponding vertex pairs
            vertex_pairs = self._find_corresponding_vertices_2d(
                mesh.vertices[boundary1_vertices],
                mesh.vertices[boundary2_vertices],
                boundary1_vertices,
                boundary2_vertices,
            )

            # Apply periodic coupling: u(vertex1) = u(vertex2)
            for vertex1_idx, vertex2_idx in vertex_pairs:
                # Master-slave approach: vertex2 = vertex1
                matrix_mod[vertex2_idx, :] = 0
                matrix_mod[vertex2_idx, vertex1_idx] = 1.0
                matrix_mod[vertex2_idx, vertex2_idx] = -1.0

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Periodic conditions typically don't modify RHS."""
        return rhs

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has proper boundary identification."""
        # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
        try:
            _ = mesh.boundary_markers
            return True
        except AttributeError:
            return False

    def _identify_boundary_vertices_for_pairing(
        self, mesh: MeshData, tolerance: float = 1e-10
    ) -> dict[int, np.ndarray]:
        """
        Identify boundary vertices for periodic pairing in 2D.

        Args:
            mesh: Mesh data
            tolerance: Geometric tolerance for boundary detection

        Returns:
            Dictionary mapping boundary IDs to vertex indices
        """
        vertices = mesh.vertices[:, :2]  # Take only x,y coordinates
        bounds = self._compute_bounding_box_2d(vertices)

        boundary_mapping = {}

        # Standard edge identification for rectangular domains
        boundary_mapping[0] = np.where(np.abs(vertices[:, 0] - bounds[0]) < tolerance)[0]  # x_min (left)
        boundary_mapping[1] = np.where(np.abs(vertices[:, 0] - bounds[1]) < tolerance)[0]  # x_max (right)
        boundary_mapping[2] = np.where(np.abs(vertices[:, 1] - bounds[2]) < tolerance)[0]  # y_min (bottom)
        boundary_mapping[3] = np.where(np.abs(vertices[:, 1] - bounds[3]) < tolerance)[0]  # y_max (top)

        return boundary_mapping

    def _find_corresponding_vertices_2d(
        self,
        boundary1_coords: np.ndarray,
        boundary2_coords: np.ndarray,
        boundary1_indices: np.ndarray,
        boundary2_indices: np.ndarray,
        tolerance: float = 1e-8,
    ) -> list[tuple[int, int]]:
        """
        Find corresponding vertex pairs between periodic boundaries in 2D.

        Args:
            boundary1_coords: Coordinates of vertices on first boundary
            boundary2_coords: Coordinates of vertices on second boundary
            boundary1_indices: Global indices of first boundary vertices
            boundary2_indices: Global indices of second boundary vertices
            tolerance: Tolerance for matching vertices

        Returns:
            List of (vertex1_idx, vertex2_idx) pairs
        """
        vertex_pairs = []

        # For rectangular domains, opposite edges should have corresponding vertices
        # that differ only in one coordinate
        for i, coord1 in enumerate(boundary1_coords[:, :2]):  # Take x,y only
            for j, coord2 in enumerate(boundary2_coords[:, :2]):
                # Check if vertices correspond (same in 1 coordinate, different in other)
                diff = np.abs(coord1 - coord2)

                # Count how many coordinates are approximately equal
                equal_coords = np.sum(diff < tolerance)

                # For 2D periodic boundaries, exactly 1 coordinate should match
                if equal_coords == 1:
                    vertex_pairs.append((boundary1_indices[i], boundary2_indices[j]))
                    break  # Each vertex on boundary1 should match exactly one on boundary2

        return vertex_pairs

    def _compute_bounding_box_2d(self, vertices: np.ndarray) -> tuple[float, float, float, float]:
        """Compute 2D bounding box of vertices."""
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (min_coords[0], max_coords[0], min_coords[1], max_coords[1])


class BoundaryConditionManager2D:
    """
    Manager for applying multiple boundary conditions to 2D MFG problems.

    Coordinates the application of different boundary condition types
    across various regions of the 2D computational domain.
    """

    def __init__(self) -> None:
        """Initialize boundary condition manager."""
        self.conditions: list[BoundaryCondition2D] = []
        self.region_map: dict[int, list[BoundaryCondition2D]] = {}

    def add_condition(self, condition: BoundaryCondition2D, boundary_region: int | str | np.ndarray) -> None:
        """
        Add boundary condition for specific region.

        Args:
            condition: Boundary condition to apply
            boundary_region: Region identifier (int), name (str), or vertex indices (array)
        """
        self.conditions.append(condition)

        if isinstance(boundary_region, int):
            if boundary_region not in self.region_map:
                self.region_map[boundary_region] = []
            self.region_map[boundary_region].append(condition)
        elif isinstance(boundary_region, str):
            # Handle named regions
            warnings.warn("Named region support not fully implemented", UserWarning)
        elif isinstance(boundary_region, np.ndarray):
            # Handle direct vertex specification
            condition._direct_vertices = boundary_region

    def identify_boundary_vertices(self, mesh: MeshData, tolerance: float = 1e-10) -> dict[str, np.ndarray]:
        """
        Identify boundary vertices for common 2D geometries.

        Args:
            mesh: Mesh data
            tolerance: Geometric tolerance for boundary detection

        Returns:
            Dictionary mapping boundary names to vertex indices
        """
        vertices = mesh.vertices[:, :2]  # Take x,y coordinates
        bounds = self._compute_bounding_box_2d(vertices)

        boundary_vertices = {}

        # Identify edge boundaries for rectangular domains
        boundary_vertices["x_min"] = np.where(np.abs(vertices[:, 0] - bounds[0]) < tolerance)[0]
        boundary_vertices["x_max"] = np.where(np.abs(vertices[:, 0] - bounds[1]) < tolerance)[0]
        boundary_vertices["y_min"] = np.where(np.abs(vertices[:, 1] - bounds[2]) < tolerance)[0]
        boundary_vertices["y_max"] = np.where(np.abs(vertices[:, 1] - bounds[3]) < tolerance)[0]

        # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
        # For non-rectangular geometries, use boundary detection from edges
        try:
            boundary_edges = mesh.boundary_edges
            if boundary_edges is not None:
                edge_vertices = self._detect_edge_vertices_from_boundary_edges(mesh)
                boundary_vertices["boundary"] = edge_vertices
        except AttributeError:
            pass

        return boundary_vertices

    def apply_all_conditions(
        self, matrix: csr_matrix, rhs: np.ndarray, mesh: MeshData, time: float = 0.0
    ) -> tuple[csr_matrix, np.ndarray]:
        """
        Apply all boundary conditions to system matrix and RHS.

        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            mesh: Mesh data
            time: Current time for time-dependent conditions

        Returns:
            Modified matrix and RHS with boundary conditions applied
        """
        matrix_mod = matrix.copy()
        rhs_mod = rhs.copy()

        # Get boundary vertex mapping
        boundary_mapping = self.identify_boundary_vertices(mesh)

        # Apply conditions by region
        for region_id, conditions in self.region_map.items():
            # Get boundary vertices for this region
            region_key = str(region_id)
            if region_key in boundary_mapping:
                boundary_indices = boundary_mapping[region_key]
            else:
                # Try to find region in standard edge names
                region_names = ["x_min", "x_max", "y_min", "y_max"]
                if region_id < len(region_names):
                    boundary_indices = boundary_mapping.get(region_names[region_id], np.array([]))
                else:
                    boundary_indices = np.array([])

            # Apply each condition for this region
            for condition in conditions:
                # Issue #543: Use try/except instead of hasattr() for optional internal attribute
                # Check for directly specified vertices
                try:
                    direct_verts = condition._direct_vertices
                    if direct_verts is not None:
                        boundary_indices = direct_verts
                except AttributeError:
                    pass

                matrix_mod = condition.apply_to_matrix(matrix_mod, mesh, boundary_indices)
                rhs_mod = condition.apply_to_rhs(rhs_mod, mesh, boundary_indices, time)

        return matrix_mod, rhs_mod

    def validate_all_conditions(self, mesh: MeshData) -> bool:
        """
        Validate all boundary conditions are compatible with mesh.

        Args:
            mesh: Mesh data to validate against

        Returns:
            True if all conditions are valid
        """
        for condition in self.conditions:
            if not condition.validate_mesh_compatibility(mesh):
                warnings.warn(f"Boundary condition {condition.name} incompatible with mesh", UserWarning)
                return False

        return True

    def _compute_bounding_box_2d(self, vertices: np.ndarray) -> tuple[float, float, float, float]:
        """Compute 2D bounding box of vertices."""
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (min_coords[0], max_coords[0], min_coords[1], max_coords[1])

    def _detect_edge_vertices_from_boundary_edges(self, mesh: MeshData) -> np.ndarray:
        """Detect boundary vertices from boundary edges."""
        # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
        try:
            boundary_edges = mesh.boundary_edges
            if boundary_edges is None:
                return np.array([])
        except AttributeError:
            return np.array([])

        # Collect all vertices that appear in boundary edges
        boundary_vertices = set()
        for edge in boundary_edges:
            for vertex_idx in edge:
                boundary_vertices.add(vertex_idx)

        return np.array(list(boundary_vertices))


class MFGBoundaryHandler2D:
    """
    Specialized boundary condition handler for Mean Field Games in 2D.

    Provides MFG-specific boundary conditions including state constraints,
    flux conservation, and optimal control boundary conditions.
    """

    def __init__(self):
        """Initialize MFG boundary handler."""
        self.hjb_manager = BoundaryConditionManager2D()
        self.fp_manager = BoundaryConditionManager2D()

    def add_state_constraint(self, region: int | str, constraint_function: Callable):
        """
        Add state constraint boundary condition for HJB equation.

        Args:
            region: Boundary region identifier
            constraint_function: Function defining state constraint
        """
        condition = DirichletBC2D(constraint_function, "State Constraint")
        self.hjb_manager.add_condition(condition, region)

    def add_no_flux_condition(self, region: int | str):
        """
        Add no-flux condition for Fokker-Planck equation.

        Args:
            region: Boundary region identifier
        """
        condition = NeumannBC2D(0.0, "No Flux")
        self.fp_manager.add_condition(condition, region)

    def add_periodic_boundary_pair(self, region1: int, region2: int):
        """
        Add periodic boundary condition between two regions.

        Args:
            region1: First boundary region
            region2: Second boundary region (paired with first)
        """
        condition = PeriodicBC2D([(region1, region2)], "Periodic Pair")
        self.hjb_manager.add_condition(condition, region1)
        self.fp_manager.add_condition(condition, region1)

    def apply_hjb_conditions(
        self, matrix: csr_matrix, rhs: np.ndarray, mesh: MeshData, time: float = 0.0
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply boundary conditions to HJB equation."""
        return self.hjb_manager.apply_all_conditions(matrix, rhs, mesh, time)

    def apply_fp_conditions(
        self, matrix: csr_matrix, rhs: np.ndarray, mesh: MeshData, time: float = 0.0
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply boundary conditions to Fokker-Planck equation."""
        return self.fp_manager.apply_all_conditions(matrix, rhs, mesh, time)

    def validate_mfg_compatibility(self, mesh: MeshData) -> bool:
        """Validate all MFG boundary conditions."""
        hjb_valid = self.hjb_manager.validate_all_conditions(mesh)
        fp_valid = self.fp_manager.validate_all_conditions(mesh)
        return hjb_valid and fp_valid


# Factory functions for common 2D boundary condition scenarios
def create_rectangle_boundary_conditions(
    domain_bounds: tuple[float, float, float, float], condition_type: str = "dirichlet_zero"
) -> BoundaryConditionManager2D:
    """
    Create standard boundary conditions for rectangular domains.

    Args:
        domain_bounds: (x_min, x_max, y_min, y_max)
        condition_type: Type of boundary condition to apply

    Returns:
        Configured boundary condition manager
    """
    manager = BoundaryConditionManager2D()

    if condition_type == "dirichlet_zero":
        # Zero Dirichlet on all edges
        for i in range(4):  # 4 edges of rectangle
            condition: DirichletBC2D | NeumannBC2D = DirichletBC2D(0.0, f"Edge_{i}")
            manager.add_condition(condition, i)

    elif condition_type == "neumann_zero":
        # Zero Neumann (no flux) on all edges
        for i in range(4):
            condition = NeumannBC2D(0.0, f"NoFlux_{i}")
            manager.add_condition(condition, i)

    elif condition_type == "periodic_x":
        # Periodic in x-direction, zero Neumann in y-direction
        periodic_condition = PeriodicBC2D([(0, 1)], "Periodic_X")  # Left-right pair
        manager.add_condition(periodic_condition, 0)
        manager.add_condition(NeumannBC2D(0.0, "NoFlux_Bottom"), 2)
        manager.add_condition(NeumannBC2D(0.0, "NoFlux_Top"), 3)

    elif condition_type == "periodic_y":
        # Periodic in y-direction, zero Neumann in x-direction
        manager.add_condition(NeumannBC2D(0.0, "NoFlux_Left"), 0)
        manager.add_condition(NeumannBC2D(0.0, "NoFlux_Right"), 1)
        periodic_condition = PeriodicBC2D([(2, 3)], "Periodic_Y")  # Bottom-top pair
        manager.add_condition(periodic_condition, 2)

    elif condition_type == "periodic_both":
        # Periodic in both directions
        periodic_x = PeriodicBC2D([(0, 1)], "Periodic_X")
        periodic_y = PeriodicBC2D([(2, 3)], "Periodic_Y")
        manager.add_condition(periodic_x, 0)
        manager.add_condition(periodic_y, 2)

    return manager


def create_circle_boundary_conditions(
    center: tuple[float, float], radius: float, condition_type: str = "dirichlet_zero"
) -> BoundaryConditionManager2D:
    """
    Create boundary conditions for circular domains.

    Args:
        center: Circle center coordinates
        radius: Circle radius
        condition_type: Type of boundary condition

    Returns:
        Configured boundary condition manager
    """
    manager = BoundaryConditionManager2D()

    def is_on_circle(x, y, t=0.0):
        """Check if point is on circle boundary."""
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        return np.abs(dist - radius) < 1e-10

    if condition_type == "dirichlet_zero":
        condition: DirichletBC2D | NeumannBC2D = DirichletBC2D(0.0, "Circle_Boundary")
    elif condition_type == "neumann_zero":
        condition = NeumannBC2D(0.0, "Circle_Boundary")
    else:
        condition = DirichletBC2D(0.0, "Circle_Boundary")

    manager.add_condition(condition, "boundary")

    return manager
