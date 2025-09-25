"""
Advanced 3D boundary condition handling for Mean Field Games.

This module provides comprehensive boundary condition support for 3D MFG problems,
including complex geometries, Robin conditions, and flux constraints.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base_geometry import MeshData


class BoundaryCondition3D(ABC):
    """
    Abstract base class for 3D boundary conditions.

    Provides the interface for implementing various boundary condition types
    including Dirichlet, Neumann, Robin, and flux constraints.
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
        # For direct vertex specification
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


class DirichletBC3D(BoundaryCondition3D):
    """
    Dirichlet boundary condition: u = g(x,y,z,t) on boundary.

    Enforces fixed values on the boundary, commonly used for
    specifying known solution values at domain boundaries.
    """

    def __init__(self, value_function: float | Callable, name: str = "Dirichlet", region_id: int | None = None):
        """
        Initialize Dirichlet boundary condition.

        Args:
            value_function: Either constant value or function(x,y,z,t) -> float
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        if callable(value_function):
            self.value_function = value_function
        else:
            self.value_function = lambda x, y, z, t: float(value_function)

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
            x, y, z = boundary_vertices[i]
            rhs_mod[idx] = self.value_function(x, y, z, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has boundary markers."""
        return hasattr(mesh, "boundary_markers") or hasattr(mesh, "vertex_markers")


class NeumannBC3D(BoundaryCondition3D):
    """
    Neumann boundary condition: ∇u·n = g(x,y,z,t) on boundary.

    Enforces flux conditions on the boundary, commonly used for
    no-flux conditions or specified gradient conditions.
    """

    def __init__(self, flux_function: float | Callable, name: str = "Neumann", region_id: int | None = None):
        """
        Initialize Neumann boundary condition.

        Args:
            flux_function: Either constant flux or function(x,y,z,t) -> float
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        if callable(flux_function):
            self.flux_function = flux_function
        else:
            self.flux_function = lambda x, y, z, t: float(flux_function)

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
        # This is a simplified implementation - would need proper surface integration
        for i, idx in enumerate(boundary_indices):
            x, y, z = boundary_vertices[i]
            # Add flux contribution (needs proper scaling by surface area)
            rhs_mod[idx] += self.flux_function(x, y, z, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has surface normal information."""
        return hasattr(mesh, "boundary_normals") or hasattr(mesh, "face_normals")


class RobinBC3D(BoundaryCondition3D):
    """
    Robin boundary condition: α·u + β·∇u·n = g(x,y,z,t) on boundary.

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
            value_function: RHS function g(x,y,z,t)
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)

        # Convert to functions if constants
        self.alpha = alpha if callable(alpha) else lambda x, y, z, t: float(alpha)
        self.beta = beta if callable(beta) else lambda x, y, z, t: float(beta)
        self.value_function = value_function if callable(value_function) else lambda x, y, z, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Robin condition to matrix."""
        matrix_mod = matrix.tolil()

        boundary_vertices = mesh.vertices[boundary_indices]

        for i, idx in enumerate(boundary_indices):
            x, y, z = boundary_vertices[i]
            alpha_val = self.alpha(x, y, z, 0.0)  # Time-independent for matrix
            self.beta(x, y, z, 0.0)

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
            x, y, z = boundary_vertices[i]
            rhs_mod[idx] += self.value_function(x, y, z, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh compatibility."""
        return True  # Robin conditions are generally compatible


class PeriodicBC3D(BoundaryCondition3D):
    """
    Periodic boundary condition: u(x1,y,z) = u(x2,y,z) for paired boundaries.

    Enforces periodicity across opposite faces of the domain,
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

        # Implementation would identify paired boundary vertices
        # and add coupling terms to matrix
        warnings.warn("Periodic BC implementation is placeholder", UserWarning)

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self, rhs: np.ndarray, mesh: MeshData, boundary_indices: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Periodic conditions typically don't modify RHS."""
        return rhs

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has proper boundary identification."""
        return hasattr(mesh, "boundary_markers")


class BoundaryConditionManager3D:
    """
    Manager for applying multiple boundary conditions to 3D MFG problems.

    Coordinates the application of different boundary condition types
    across various regions of the computational domain.
    """

    def __init__(self) -> None:
        """Initialize boundary condition manager."""
        self.conditions: list[BoundaryCondition3D] = []
        self.region_map: dict[int, list[BoundaryCondition3D]] = {}

    def add_condition(self, condition: BoundaryCondition3D, boundary_region: int | str | np.ndarray) -> None:
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
            # Handle named regions (would need mesh metadata)
            warnings.warn("Named region support not fully implemented", UserWarning)
        elif isinstance(boundary_region, np.ndarray):
            # Handle direct vertex specification
            condition._direct_vertices = boundary_region

    def identify_boundary_vertices(self, mesh: MeshData, tolerance: float = 1e-10) -> dict[str, np.ndarray]:
        """
        Identify boundary vertices for common 3D geometries.

        Args:
            mesh: Mesh data
            tolerance: Geometric tolerance for boundary detection

        Returns:
            Dictionary mapping boundary names to vertex indices
        """
        vertices = mesh.vertices
        bounds = self._compute_bounding_box(vertices)

        boundary_vertices = {}

        # Identify face boundaries for box domains
        boundary_vertices["x_min"] = np.where(np.abs(vertices[:, 0] - bounds[0]) < tolerance)[0]
        boundary_vertices["x_max"] = np.where(np.abs(vertices[:, 0] - bounds[1]) < tolerance)[0]
        boundary_vertices["y_min"] = np.where(np.abs(vertices[:, 1] - bounds[2]) < tolerance)[0]
        boundary_vertices["y_max"] = np.where(np.abs(vertices[:, 1] - bounds[3]) < tolerance)[0]
        boundary_vertices["z_min"] = np.where(np.abs(vertices[:, 2] - bounds[4]) < tolerance)[0]
        boundary_vertices["z_max"] = np.where(np.abs(vertices[:, 2] - bounds[5]) < tolerance)[0]

        # For non-box geometries, use surface detection from boundary_faces
        if hasattr(mesh, "boundary_faces") and mesh.boundary_faces is not None:
            surface_vertices = self._detect_surface_vertices_from_boundary_faces(mesh)
            boundary_vertices["surface"] = surface_vertices

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
            if region_id in boundary_mapping:
                boundary_indices = boundary_mapping[region_id]
            else:
                # Try to find region in standard face names
                region_names = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
                if region_id < len(region_names):
                    boundary_indices = boundary_mapping.get(region_names[region_id], np.array([]))
                else:
                    boundary_indices = np.array([])

            # Apply each condition for this region
            for condition in conditions:
                # Check for directly specified vertices
                if hasattr(condition, "_direct_vertices"):
                    boundary_indices = condition._direct_vertices

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

    def _compute_bounding_box(self, vertices: np.ndarray) -> tuple[float, ...]:
        """Compute bounding box of vertices."""
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2])

    def _detect_surface_vertices_from_boundary_faces(self, mesh: MeshData) -> np.ndarray:
        """Detect surface vertices from boundary faces."""
        if not hasattr(mesh, "boundary_faces") or mesh.boundary_faces is None:
            return np.array([])

        # Collect all vertices that appear in boundary faces
        surface_vertices = set()
        for face in mesh.boundary_faces:
            for vertex_idx in face:
                surface_vertices.add(vertex_idx)

        return np.array(list(surface_vertices))


class MFGBoundaryHandler3D:
    """
    Specialized boundary condition handler for Mean Field Games in 3D.

    Provides MFG-specific boundary conditions including state constraints,
    flux conservation, and optimal control boundary conditions.
    """

    def __init__(self):
        """Initialize MFG boundary handler."""
        self.hjb_manager = BoundaryConditionManager3D()
        self.fp_manager = BoundaryConditionManager3D()

    def add_state_constraint(self, region: int | str, constraint_function: Callable):
        """
        Add state constraint boundary condition for HJB equation.

        Args:
            region: Boundary region identifier
            constraint_function: Function defining state constraint
        """
        condition = DirichletBC3D(constraint_function, "State Constraint")
        self.hjb_manager.add_condition(condition, region)

    def add_no_flux_condition(self, region: int | str):
        """
        Add no-flux condition for Fokker-Planck equation.

        Args:
            region: Boundary region identifier
        """
        condition = NeumannBC3D(0.0, "No Flux")
        self.fp_manager.add_condition(condition, region)

    def add_mass_conservation(self, region: int | str, inflow_rate: float | Callable):
        """
        Add mass conservation boundary condition.

        Args:
            region: Boundary region identifier
            inflow_rate: Mass inflow rate function
        """
        condition = NeumannBC3D(inflow_rate, "Mass Conservation")
        self.fp_manager.add_condition(condition, region)

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


# Factory functions for common boundary condition scenarios
def create_box_boundary_conditions(
    domain_bounds: tuple[float, float, float, float, float, float], condition_type: str = "dirichlet_zero"
) -> BoundaryConditionManager3D:
    """
    Create standard boundary conditions for box domains.

    Args:
        domain_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        condition_type: Type of boundary condition to apply

    Returns:
        Configured boundary condition manager
    """
    manager = BoundaryConditionManager3D()

    if condition_type == "dirichlet_zero":
        # Zero Dirichlet on all faces
        for i in range(6):  # 6 faces of box
            condition = DirichletBC3D(0.0, f"Face_{i}")
            manager.add_condition(condition, i)

    elif condition_type == "neumann_zero":
        # Zero Neumann (no flux) on all faces
        for i in range(6):
            condition = NeumannBC3D(0.0, f"NoFlux_{i}")
            manager.add_condition(condition, i)

    elif condition_type == "mixed":
        # Mixed conditions: Dirichlet on x-faces, Neumann on y,z-faces
        manager.add_condition(DirichletBC3D(0.0, "X_Min"), 0)  # x_min
        manager.add_condition(DirichletBC3D(0.0, "X_Max"), 1)  # x_max
        for i in range(2, 6):  # y and z faces
            manager.add_condition(NeumannBC3D(0.0, f"NoFlux_{i}"), i)

    return manager


def create_sphere_boundary_conditions(
    center: tuple[float, float, float], radius: float, condition_type: str = "dirichlet_zero"
) -> BoundaryConditionManager3D:
    """
    Create boundary conditions for spherical domains.

    Args:
        center: Sphere center coordinates
        radius: Sphere radius
        condition_type: Type of boundary condition

    Returns:
        Configured boundary condition manager
    """
    manager = BoundaryConditionManager3D()

    def is_on_sphere(x, y, z, t=0.0):
        """Check if point is on sphere boundary."""
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        return np.abs(dist - radius) < 1e-10

    if condition_type == "dirichlet_zero":
        condition = DirichletBC3D(0.0, "Sphere_Surface")
    elif condition_type == "neumann_zero":
        condition = NeumannBC3D(0.0, "Sphere_Surface")
    else:
        condition = DirichletBC3D(0.0, "Sphere_Surface")

    manager.add_condition(condition, "surface")

    return manager
