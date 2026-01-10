"""
FEM/mesh-based boundary condition application.

This module provides a unified interface for FEM boundary conditions while
maintaining dimension-specific optimized implementations:

- **1D**: Uses optimized classes from fem_bc_1d.py (point-based boundaries)
- **2D**: Uses optimized classes from fem_bc_2d.py (edge-based boundaries)
- **3D**: Uses optimized classes from fem_bc_3d.py (face-based boundaries)

The FEMApplicator class provides dimension dispatch, while the underlying
implementations are kept separate for performance.

Design:
- BC Specification: Uses unified BoundaryConditions from conditions.py
- BC Application: Dimension-specific (1D points vs 2D edges vs 3D faces)

Exports:
- FEMApplicator: Class-based dispatcher for FEM BC application
- 1D classes: DirichletBC1D, NeumannBC1D, RobinBC1D, PeriodicBC1D, etc.
- 2D classes: DirichletBC2D, NeumannBC2D, RobinBC2D, PeriodicBC2D, etc.
- 3D classes: DirichletBC3D, NeumannBC3D, RobinBC3D, PeriodicBC3D, etc.
- BoundaryManager: Manager for complex geometry domains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Import base class for inheritance
from .applicator_base import BaseUnstructuredApplicator

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.sparse import csr_matrix

    from mfg_pde.geometry.meshes.mesh_data import MeshData

# =============================================================================
# Import dimension-specific implementations
# =============================================================================

# 1D implementations (point-based boundaries)
from .fem_bc_1d import (
    BoundaryCondition1D,
    BoundaryConditionManager1D,
    DirichletBC1D,
    MFGBoundaryHandler1D,
    NeumannBC1D,
    PeriodicBC1D,
    RobinBC1D,
    create_interval_boundary_conditions,
)

# 2D implementations (edge-based boundaries)
from .fem_bc_2d import (
    BoundaryCondition2D,
    BoundaryConditionManager2D,
    DirichletBC2D,
    MFGBoundaryHandler2D,
    NeumannBC2D,
    PeriodicBC2D,
    RobinBC2D,
    create_circle_boundary_conditions,
    create_rectangle_boundary_conditions,
)

# 3D implementations (face-based boundaries)
from .fem_bc_3d import (
    BoundaryCondition3D,
    BoundaryConditionManager3D,
    DirichletBC3D,
    MFGBoundaryHandler3D,
    NeumannBC3D,
    PeriodicBC3D,
    RobinBC3D,
    create_box_boundary_conditions,
    create_sphere_boundary_conditions,
)

# =============================================================================
# Unified FEM Applicator (dispatches to dimension-specific implementations)
# =============================================================================


class FEMApplicator(BaseUnstructuredApplicator):
    """
    Finite Element Method boundary condition applicator.

    Provides a unified interface while dispatching to dimension-specific
    optimized implementations:
    - 1D: Point-based boundary handling (fem_bc_1d.py)
    - 2D: Edge-based boundary handling (fem_bc_2d.py)
    - 3D: Face-based boundary handling (fem_bc_3d.py)

    Inherits from BaseUnstructuredApplicator for consistent interface across
    all BC applicators.

    Usage:
        # Create 2D applicator
        applicator = FEMApplicator(dimension=2)

        # Add conditions
        applicator.add_dirichlet(region=0, value=0.0)
        applicator.add_neumann(region=1, flux=0.0)

        # Apply to system
        matrix, rhs = applicator.apply(matrix, rhs, mesh, time=0.0)

        # Or use dimension-specific managers directly
        manager_2d = applicator.get_manager()  # Returns BoundaryConditionManager2D
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize FEM applicator.

        Args:
            dimension: Spatial dimension (1, 2, or 3)
        """
        super().__init__(dimension)  # Validates dimension in base class

        # Create dimension-specific manager
        if dimension == 1:
            self._manager: BoundaryConditionManager1D | BoundaryConditionManager2D | BoundaryConditionManager3D = (
                BoundaryConditionManager1D()
            )
        elif dimension == 2:
            self._manager = BoundaryConditionManager2D()
        else:
            self._manager = BoundaryConditionManager3D()

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dimension

    def get_manager(
        self,
    ) -> BoundaryConditionManager1D | BoundaryConditionManager2D | BoundaryConditionManager3D:
        """Get the underlying dimension-specific manager."""
        return self._manager

    def add_dirichlet(
        self,
        region: int | str | np.ndarray,
        value: float | Callable = 0.0,
        name: str = "Dirichlet",
    ) -> None:
        """Add Dirichlet BC to a region."""
        if self._dimension == 1:
            condition = DirichletBC1D(value, name)
        elif self._dimension == 2:
            condition = DirichletBC2D(value, name)
        else:
            condition = DirichletBC3D(value, name)
        self._manager.add_condition(condition, region)

    def add_neumann(
        self,
        region: int | str | np.ndarray,
        flux: float | Callable = 0.0,
        name: str = "Neumann",
    ) -> None:
        """Add Neumann BC to a region."""
        if self._dimension == 1:
            condition = NeumannBC1D(flux, name)
        elif self._dimension == 2:
            condition = NeumannBC2D(flux, name)
        else:
            condition = NeumannBC3D(flux, name)
        self._manager.add_condition(condition, region)

    def add_robin(
        self,
        region: int | str | np.ndarray,
        alpha: float | Callable,
        beta: float | Callable,
        value: float | Callable = 0.0,
        name: str = "Robin",
    ) -> None:
        """Add Robin BC to a region."""
        if self._dimension == 1:
            condition = RobinBC1D(alpha, beta, value, name)
        elif self._dimension == 2:
            condition = RobinBC2D(alpha, beta, value, name)
        else:
            condition = RobinBC3D(alpha, beta, value, name)
        self._manager.add_condition(condition, region)

    def add_periodic(
        self,
        paired_boundaries: list[tuple[int, int]] | np.ndarray | None = None,
        name: str = "Periodic",
    ) -> None:
        """Add periodic BC between boundary pairs."""
        if self._dimension == 1:
            # In 1D, periodic links left and right endpoints
            condition = PeriodicBC1D(name)
            # Use array of endpoint indices if not provided
            indices = paired_boundaries if paired_boundaries is not None else np.array([0, -1])
            self._manager.add_condition(condition, indices)
        elif self._dimension == 2:
            condition = PeriodicBC2D(paired_boundaries, name)
            # Add to first region in each pair
            for region1, _ in paired_boundaries:
                self._manager.add_condition(condition, region1)
        else:
            condition = PeriodicBC3D(paired_boundaries, name)
            # Add to first region in each pair
            for region1, _ in paired_boundaries:
                self._manager.add_condition(condition, region1)

    def apply(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """
        Apply all boundary conditions to system.

        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            mesh: Mesh data
            time: Current time for time-dependent BCs

        Returns:
            Modified (matrix, rhs) tuple
        """
        return self._manager.apply_all_conditions(matrix, rhs, mesh, time)

    def validate(self, mesh: MeshData) -> bool:
        """Validate all BCs are compatible with mesh."""
        return self._manager.validate_all_conditions(mesh)


# =============================================================================
# MFG-Specific FEM Handler (unified interface)
# =============================================================================


class MFGBoundaryHandlerFEM:
    """
    MFG-specific boundary handler with unified interface.

    Dispatches to dimension-specific MFGBoundaryHandler2D or MFGBoundaryHandler3D.
    """

    def __init__(self, dimension: int = 2):
        """Initialize MFG boundary handler."""
        if dimension not in (2, 3):
            raise ValueError("MFGBoundaryHandlerFEM only supports 2D and 3D")

        self._dimension = dimension

        if dimension == 2:
            self._handler: MFGBoundaryHandler2D | MFGBoundaryHandler3D = MFGBoundaryHandler2D()
        else:
            self._handler = MFGBoundaryHandler3D()

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_handler(self) -> MFGBoundaryHandler2D | MFGBoundaryHandler3D:
        """Get underlying dimension-specific handler."""
        return self._handler

    def add_state_constraint(self, region: int | str, constraint_function: Callable) -> None:
        """Add state constraint for HJB equation."""
        self._handler.add_state_constraint(region, constraint_function)

    def add_no_flux_condition(self, region: int | str) -> None:
        """Add no-flux condition for FP equation."""
        self._handler.add_no_flux_condition(region)

    def add_periodic_boundary_pair(self, region1: int, region2: int) -> None:
        """Add periodic BC between regions."""
        self._handler.add_periodic_boundary_pair(region1, region2)

    def apply_hjb_conditions(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply BCs to HJB equation."""
        return self._handler.apply_hjb_conditions(matrix, rhs, mesh, time)

    def apply_fp_conditions(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply BCs to Fokker-Planck equation."""
        return self._handler.apply_fp_conditions(matrix, rhs, mesh, time)

    def validate_mfg_compatibility(self, mesh: MeshData) -> bool:
        """Validate MFG BC compatibility."""
        return self._handler.validate_mfg_compatibility(mesh)


# =============================================================================
# Geometric Boundary Condition (from bc_manager.py)
# =============================================================================


@dataclass
class GeometricBoundaryCondition:
    """
    Enhanced boundary condition for complex geometry domains.

    Extends basic BCs to handle multiple boundary regions,
    curved boundaries, and spatially-varying conditions.
    """

    region_id: int
    bc_type: str  # 'dirichlet', 'neumann', 'robin', 'no_flux'

    value: float | Callable[[np.ndarray], np.ndarray] = None
    gradient_value: float | Callable[[np.ndarray], np.ndarray] = None

    # Robin BC parameters
    alpha: float | Callable[[np.ndarray], np.ndarray] = None
    beta: float | Callable[[np.ndarray], np.ndarray] = None

    time_dependent: bool = False
    spatial_function: Callable[[np.ndarray, float], np.ndarray] | None = None

    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate BC parameters."""
        if self.bc_type == "robin":
            if self.alpha is None or self.beta is None:
                raise ValueError("Robin boundary conditions require alpha and beta")

        if self.bc_type == "dirichlet" and self.value is None:
            raise ValueError("Dirichlet boundary condition requires value")

        if self.bc_type == "neumann" and self.gradient_value is None and self.value is None:
            raise ValueError("Neumann boundary condition requires gradient_value or value")

    def evaluate(self, coordinates: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Evaluate BC at coordinates and time."""
        if self.time_dependent and self.spatial_function is not None:
            return self.spatial_function(coordinates, time)
        elif callable(self.value):
            return self.value(coordinates)
        else:
            return np.full(coordinates.shape[0], self.value)

    def evaluate_gradient(self, coordinates: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Evaluate gradient BC."""
        if callable(self.gradient_value):
            return self.gradient_value(coordinates)
        else:
            return np.full(coordinates.shape[0], self.gradient_value)


class BoundaryManager:
    """
    Manager for boundary conditions on complex geometric domains.

    Handles mapping between geometric boundary regions and BC specifications.
    """

    def __init__(self, mesh_data: MeshData):
        """Initialize boundary manager."""
        self.mesh_data = mesh_data
        self.boundary_conditions: dict[int, GeometricBoundaryCondition] = {}
        self.boundary_nodes: dict[int, np.ndarray] = {}
        self.boundary_faces: dict[int, np.ndarray] = {}

        self._extract_boundary_regions()

    def _extract_boundary_regions(self):
        """Extract boundary regions from mesh."""
        unique_regions = np.unique(self.mesh_data.boundary_tags)

        for region_id in unique_regions:
            if region_id == 0:
                continue

            region_mask = self.mesh_data.boundary_tags == region_id
            region_nodes = np.where(region_mask)[0]
            self.boundary_nodes[region_id] = region_nodes

            # Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
            try:
                boundary_faces = self.mesh_data.boundary_faces
                if boundary_faces is not None:
                    region_faces = []
                    for face in boundary_faces:
                        if all(node in region_nodes for node in face):
                            region_faces.append(face)
                    self.boundary_faces[region_id] = np.array(region_faces)
            except AttributeError:
                pass

    def add_boundary_condition(
        self,
        region_id: int,
        bc_type: str,
        value: float | Callable | None = None,
        **kwargs,
    ) -> GeometricBoundaryCondition:
        """Add BC for a region."""
        bc = GeometricBoundaryCondition(region_id=region_id, bc_type=bc_type, value=value, **kwargs)
        self.boundary_conditions[region_id] = bc
        return bc

    def get_boundary_nodes(self, region_id: int) -> np.ndarray:
        """Get node indices for a region."""
        return self.boundary_nodes.get(region_id, np.array([], dtype=int))

    def get_boundary_coordinates(self, region_id: int) -> np.ndarray:
        """Get coordinates of boundary nodes."""
        node_indices = self.get_boundary_nodes(region_id)
        return self.mesh_data.vertices[node_indices]

    def evaluate_boundary_condition(self, region_id: int, time: float = 0.0) -> np.ndarray:
        """Evaluate BC values for a region."""
        if region_id not in self.boundary_conditions:
            raise ValueError(f"No boundary condition defined for region {region_id}")

        bc = self.boundary_conditions[region_id]
        coordinates = self.get_boundary_coordinates(region_id)
        return bc.evaluate(coordinates, time)

    def apply_dirichlet_conditions(
        self,
        system_matrix: np.ndarray,
        rhs_vector: np.ndarray,
        time: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Dirichlet BCs to linear system."""
        for region_id, bc in self.boundary_conditions.items():
            if bc.bc_type != "dirichlet":
                continue

            boundary_nodes = self.get_boundary_nodes(region_id)
            if len(boundary_nodes) == 0:
                continue

            boundary_values = self.evaluate_boundary_condition(region_id, time)

            for i, node_idx in enumerate(boundary_nodes):
                system_matrix[node_idx, :] = 0.0
                system_matrix[node_idx, node_idx] = 1.0
                rhs_vector[node_idx] = boundary_values[i]

        return system_matrix, rhs_vector

    def apply_neumann_conditions(self, rhs_vector: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Apply Neumann BCs to RHS."""
        for region_id, bc in self.boundary_conditions.items():
            if bc.bc_type != "neumann":
                continue

            boundary_nodes = self.get_boundary_nodes(region_id)
            if len(boundary_nodes) == 0:
                continue

            gradient_values = bc.evaluate_gradient(self.get_boundary_coordinates(region_id), time)

            for i, node_idx in enumerate(boundary_nodes):
                rhs_vector[node_idx] += gradient_values[i]

        return rhs_vector

    def create_legacy_boundary_conditions(self):
        """Create 1D FDM BoundaryConditions for backward compatibility."""
        from .fdm_bc_1d import BoundaryConditions

        left_bc = None
        right_bc = None
        bc_type = "periodic"

        for region_id, bc in self.boundary_conditions.items():
            if region_id == 1:
                left_bc = bc
                bc_type = bc.bc_type
            elif region_id == 2:
                right_bc = bc

        left_value = None
        right_value = None

        if left_bc and not callable(left_bc.value):
            left_value = left_bc.value
        if right_bc and not callable(right_bc.value):
            right_value = right_bc.value

        return BoundaryConditions(type=bc_type, left_value=left_value, right_value=right_value)

    def get_summary(self) -> dict[str, Any]:
        """Get BC setup summary."""
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


# =============================================================================
# Dimension-agnostic aliases (for generic code)
# =============================================================================

# These allow code to work with either 2D or 3D by checking dimension
BoundaryConditionFEM = BoundaryCondition2D  # Default to 2D
BoundaryConditionManagerFEM = BoundaryConditionManager2D  # Default to 2D


def get_bc_class(dimension: int, bc_type: str):
    """Get appropriate BC class for dimension and type."""
    if dimension == 1:
        classes = {
            "dirichlet": DirichletBC1D,
            "neumann": NeumannBC1D,
            "robin": RobinBC1D,
            "periodic": PeriodicBC1D,
        }
    elif dimension == 2:
        classes = {
            "dirichlet": DirichletBC2D,
            "neumann": NeumannBC2D,
            "robin": RobinBC2D,
            "periodic": PeriodicBC2D,
        }
    elif dimension == 3:
        classes = {
            "dirichlet": DirichletBC3D,
            "neumann": NeumannBC3D,
            "robin": RobinBC3D,
            "periodic": PeriodicBC3D,
        }
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    if bc_type.lower() not in classes:
        raise ValueError(f"Unknown BC type: {bc_type}")

    return classes[bc_type.lower()]


def get_manager_class(dimension: int):
    """Get appropriate manager class for dimension."""
    if dimension == 1:
        return BoundaryConditionManager1D
    elif dimension == 2:
        return BoundaryConditionManager2D
    elif dimension == 3:
        return BoundaryConditionManager3D
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


__all__ = [
    # Unified dispatchers
    "FEMApplicator",
    "MFGBoundaryHandlerFEM",
    # Geometric BC manager
    "BoundaryManager",
    "GeometricBoundaryCondition",
    # 1D classes (optimized, point-based boundaries)
    "BoundaryCondition1D",
    "BoundaryConditionManager1D",
    "DirichletBC1D",
    "NeumannBC1D",
    "RobinBC1D",
    "PeriodicBC1D",
    "MFGBoundaryHandler1D",
    "create_interval_boundary_conditions",
    # 2D classes (optimized, edge-based boundaries)
    "BoundaryCondition2D",
    "BoundaryConditionManager2D",
    "DirichletBC2D",
    "NeumannBC2D",
    "RobinBC2D",
    "PeriodicBC2D",
    "MFGBoundaryHandler2D",
    "create_rectangle_boundary_conditions",
    "create_circle_boundary_conditions",
    # 3D classes (optimized, face-based boundaries)
    "BoundaryCondition3D",
    "BoundaryConditionManager3D",
    "DirichletBC3D",
    "NeumannBC3D",
    "RobinBC3D",
    "PeriodicBC3D",
    "MFGBoundaryHandler3D",
    "create_box_boundary_conditions",
    "create_sphere_boundary_conditions",
    # Dimension-agnostic aliases
    "BoundaryConditionFEM",
    "BoundaryConditionManagerFEM",
    # Helper functions
    "get_bc_class",
    "get_manager_class",
]
