"""
Boundary condition management for complex geometry domains.

This module extends the existing boundary condition system to handle
complex 2D/3D geometries with multiple boundary regions, curved boundaries,
and advanced boundary condition types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .domain_1d import BoundaryConditions

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base_geometry import MeshData


@dataclass
class GeometricBoundaryCondition:
    """
    Enhanced boundary condition for complex geometry domains.

    Extends the basic BoundaryConditions to handle multiple boundary regions,
    curved boundaries, and spatially-varying boundary conditions.
    """

    # Basic boundary condition properties
    region_id: int  # Boundary region identifier
    bc_type: str  # 'dirichlet', 'neumann', 'robin', 'no_flux'

    # Values (can be constants or functions)
    value: float | Callable[[np.ndarray], np.ndarray] = None
    gradient_value: float | Callable[[np.ndarray], np.ndarray] = None

    # Robin boundary condition parameters
    alpha: float | Callable[[np.ndarray], np.ndarray] = None  # coefficient of u
    beta: float | Callable[[np.ndarray], np.ndarray] = None  # coefficient of du/dn

    # Spatial and temporal dependencies
    time_dependent: bool = False
    spatial_function: Callable[[np.ndarray, float], np.ndarray] | None = None

    # Metadata
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate boundary condition parameters."""
        if self.bc_type == "robin":
            if self.alpha is None or self.beta is None:
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")

        if self.bc_type in ["dirichlet", "neumann"] and self.value is None:
            raise ValueError(f"{self.bc_type} boundary condition requires value")

    def evaluate(self, coordinates: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Evaluate boundary condition at given coordinates and time.

        Args:
            coordinates: Boundary point coordinates (N, dim)
            time: Current time

        Returns:
            Boundary condition values at coordinates
        """

        if self.time_dependent and self.spatial_function is not None:
            return self.spatial_function(coordinates, time)
        elif callable(self.value):
            return self.value(coordinates)
        else:
            return np.full(coordinates.shape[0], self.value)

    def evaluate_gradient(self, coordinates: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Evaluate gradient boundary condition (for Neumann/Robin)."""
        if callable(self.gradient_value):
            return self.gradient_value(coordinates)
        else:
            return np.full(coordinates.shape[0], self.gradient_value)


class BoundaryManager:
    """
    Manager for boundary conditions on complex geometric domains.

    This class handles the mapping between geometric boundary regions
    and boundary condition specifications, supporting multiple regions
    with different boundary condition types.
    """

    def __init__(self, mesh_data: MeshData):
        """
        Initialize boundary manager.

        Args:
            mesh_data: Mesh data containing boundary information
        """
        self.mesh_data = mesh_data
        self.boundary_conditions: dict[int, GeometricBoundaryCondition] = {}
        self.boundary_nodes: dict[int, np.ndarray] = {}
        self.boundary_faces: dict[int, np.ndarray] = {}

        # Extract boundary information from mesh
        self._extract_boundary_regions()

    def _extract_boundary_regions(self):
        """Extract boundary regions from mesh data."""
        # Group boundary faces by region
        unique_regions = np.unique(self.mesh_data.boundary_tags)

        for region_id in unique_regions:
            if region_id == 0:  # Skip interior nodes
                continue

            # Find nodes belonging to this boundary region
            region_mask = self.mesh_data.boundary_tags == region_id
            region_nodes = np.where(region_mask)[0]
            self.boundary_nodes[region_id] = region_nodes

            # Find boundary faces in this region
            if hasattr(self.mesh_data, "boundary_faces") and self.mesh_data.boundary_faces is not None:
                # For 2D: boundary faces are edges
                region_faces = []
                for face in self.mesh_data.boundary_faces:
                    if all(node in region_nodes for node in face):
                        region_faces.append(face)
                self.boundary_faces[region_id] = np.array(region_faces)

    def add_boundary_condition(
        self,
        region_id: int,
        bc_type: str,
        value: float | Callable | None = None,
        **kwargs,
    ) -> GeometricBoundaryCondition:
        """
        Add boundary condition for a specific region.

        Args:
            region_id: Boundary region identifier
            bc_type: Boundary condition type
            value: Boundary condition value (constant or function)
            **kwargs: Additional boundary condition parameters

        Returns:
            Created boundary condition object
        """

        bc = GeometricBoundaryCondition(region_id=region_id, bc_type=bc_type, value=value, **kwargs)

        self.boundary_conditions[region_id] = bc
        return bc

    def get_boundary_nodes(self, region_id: int) -> np.ndarray:
        """Get node indices for a boundary region."""
        return self.boundary_nodes.get(region_id, np.array([], dtype=int))

    def get_boundary_coordinates(self, region_id: int) -> np.ndarray:
        """Get coordinates of boundary nodes for a region."""
        node_indices = self.get_boundary_nodes(region_id)
        return self.mesh_data.vertices[node_indices]

    def evaluate_boundary_condition(self, region_id: int, time: float = 0.0) -> np.ndarray:
        """
        Evaluate boundary condition values for a region.

        Args:
            region_id: Boundary region identifier
            time: Current time

        Returns:
            Boundary condition values at boundary nodes
        """

        if region_id not in self.boundary_conditions:
            raise ValueError(f"No boundary condition defined for region {region_id}")

        bc = self.boundary_conditions[region_id]
        coordinates = self.get_boundary_coordinates(region_id)

        return bc.evaluate(coordinates, time)

    def apply_dirichlet_conditions(
        self, system_matrix: np.ndarray, rhs_vector: np.ndarray, time: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Dirichlet boundary conditions to linear system.

        Args:
            system_matrix: System matrix (will be modified)
            rhs_vector: Right-hand side vector (will be modified)
            time: Current time

        Returns:
            Modified system matrix and RHS vector
        """

        for region_id, bc in self.boundary_conditions.items():
            if bc.bc_type != "dirichlet":
                continue

            # Get boundary nodes
            boundary_nodes = self.get_boundary_nodes(region_id)
            if len(boundary_nodes) == 0:
                continue

            # Get boundary values
            boundary_values = self.evaluate_boundary_condition(region_id, time)

            # Apply Dirichlet conditions: set row to identity, RHS to boundary value
            for i, node_idx in enumerate(boundary_nodes):
                # Clear row
                system_matrix[node_idx, :] = 0.0
                # Set diagonal to 1
                system_matrix[node_idx, node_idx] = 1.0
                # Set RHS to boundary value
                rhs_vector[node_idx] = boundary_values[i]

        return system_matrix, rhs_vector

    def apply_neumann_conditions(self, rhs_vector: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Apply Neumann boundary conditions to RHS vector.

        Args:
            rhs_vector: Right-hand side vector (will be modified)
            time: Current time

        Returns:
            Modified RHS vector
        """

        for region_id, bc in self.boundary_conditions.items():
            if bc.bc_type != "neumann":
                continue

            # Get boundary information
            boundary_nodes = self.get_boundary_nodes(region_id)
            if len(boundary_nodes) == 0:
                continue

            # Get gradient values
            gradient_values = bc.evaluate_gradient(self.get_boundary_coordinates(region_id), time)

            # Apply Neumann conditions to RHS
            # This is a simplified implementation - full FEM would require
            # integration over boundary elements
            for i, node_idx in enumerate(boundary_nodes):
                rhs_vector[node_idx] += gradient_values[i]

        return rhs_vector

    def create_legacy_boundary_conditions(self) -> BoundaryConditions:
        """
        Create legacy BoundaryConditions object for backward compatibility.

        This method maps the new geometric boundary conditions back to the
        original 1D boundary condition format for compatibility with existing solvers.
        """

        # Find boundary conditions for left and right boundaries (for 1D compatibility)
        left_bc = None
        right_bc = None
        bc_type = "periodic"  # Default

        for region_id, bc in self.boundary_conditions.items():
            # Map geometric regions to left/right for 1D compatibility
            if region_id == 1:  # Assume region 1 is left boundary
                left_bc = bc
                bc_type = bc.bc_type
            elif region_id == 2:  # Assume region 2 is right boundary
                right_bc = bc

        # Extract values
        left_value = None
        right_value = None

        if left_bc and not callable(left_bc.value):
            left_value = left_bc.value
        if right_bc and not callable(right_bc.value):
            right_value = right_bc.value

        return BoundaryConditions(type=bc_type, left_value=left_value, right_value=right_value)

    def visualize_boundary_conditions(self):
        """Visualize boundary conditions using PyVista."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista is required for boundary visualization")

        # Create PyVista mesh
        mesh = self.mesh_data.to_pyvista()

        # Add boundary condition information
        bc_data = np.zeros(self.mesh_data.num_vertices)

        for region_id, bc in self.boundary_conditions.items():
            boundary_nodes = self.get_boundary_nodes(region_id)
            bc_data[boundary_nodes] = region_id

        mesh.point_data["boundary_regions"] = bc_data

        # Create plotter
        plotter = pv.Plotter(title="Boundary Conditions")
        plotter.add_mesh(mesh, scalars="boundary_regions", show_edges=True)

        # Add text annotations for each boundary condition
        for region_id, bc in self.boundary_conditions.items():
            coordinates = self.get_boundary_coordinates(region_id)
            if len(coordinates) > 0:
                center = np.mean(coordinates, axis=0)
                if self.mesh_data.dimension == 2:
                    # Add 2D text annotation
                    plotter.add_point_labels([center], [f"Region {region_id}: {bc.bc_type}"], point_size=0)

        plotter.show()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of boundary condition setup."""
        summary: dict[str, Any] = {"num_regions": len(self.boundary_conditions), "regions": {}}

        for region_id, bc in self.boundary_conditions.items():
            num_nodes = len(self.get_boundary_nodes(region_id))
            summary["regions"][region_id] = {
                "type": bc.bc_type,
                "num_nodes": num_nodes,
                "description": bc.description,
                "time_dependent": bc.time_dependent,
            }

        return summary
