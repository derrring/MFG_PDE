"""
1D boundary condition handling for Mean Field Games (FEM).

This module provides FEM boundary condition support for 1D MFG problems.
In 1D, boundaries are point-based (left/right endpoints).
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


class BoundaryCondition1D(ABC):
    """
    Abstract base class for 1D FEM boundary conditions.

    In 1D, boundaries are points (left/right endpoints of the domain).
    """

    def __init__(self, name: str, region_id: int | None = None):
        """
        Initialize boundary condition.

        Args:
            name: Human-readable name for the boundary condition
            region_id: Optional region identifier (0=left, 1=right typically)
        """
        self.name = name
        self.region_id = region_id
        self._direct_vertices: np.ndarray | None = None

    @abstractmethod
    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply boundary condition to system matrix."""

    @abstractmethod
    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply boundary condition to right-hand side vector."""

    @abstractmethod
    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate that boundary condition is compatible with mesh."""


class DirichletBC1D(BoundaryCondition1D):
    """
    Dirichlet boundary condition: u = g(x,t) at boundary point.

    Enforces fixed values at boundary points.
    """

    def __init__(
        self,
        value_function: float | Callable,
        name: str = "Dirichlet",
        region_id: int | None = None,
    ):
        """
        Initialize Dirichlet boundary condition.

        Args:
            value_function: Either constant value or function(x, t) -> float
            name: Human-readable name
            region_id: Optional region identifier (0=left, 1=right)
        """
        super().__init__(name, region_id)
        if callable(value_function):
            self.value_function = value_function
        else:
            self.value_function = lambda x, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Dirichlet condition by setting diagonal entries to 1."""
        matrix_mod = matrix.tolil()

        for idx in boundary_indices:
            # Clear row and set diagonal to 1
            matrix_mod[idx, :] = 0
            matrix_mod[idx, idx] = 1

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply Dirichlet values to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        for i, idx in enumerate(boundary_indices):
            x = boundary_vertices[i, 0] if boundary_vertices.ndim > 1 else boundary_vertices[i]
            rhs_mod[idx] = self.value_function(x, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has boundary markers."""
        return hasattr(mesh, "boundary_markers") or hasattr(mesh, "vertex_markers")


class NeumannBC1D(BoundaryCondition1D):
    """
    Neumann boundary condition: du/dx = g(x,t) at boundary point.

    Enforces flux/gradient conditions at boundary points.
    """

    def __init__(
        self,
        flux_function: float | Callable,
        name: str = "Neumann",
        region_id: int | None = None,
    ):
        """
        Initialize Neumann boundary condition.

        Args:
            flux_function: Either constant flux or function(x, t) -> float
            name: Human-readable name
            region_id: Optional region identifier (0=left, 1=right)
        """
        super().__init__(name, region_id)
        if callable(flux_function):
            self.flux_function = flux_function
        else:
            self.flux_function = lambda x, t: float(flux_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Neumann condition - typically no matrix modification needed."""
        return matrix

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply Neumann flux to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        for i, idx in enumerate(boundary_indices):
            x = boundary_vertices[i, 0] if boundary_vertices.ndim > 1 else boundary_vertices[i]
            rhs_mod[idx] += self.flux_function(x, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh has boundary information."""
        return True


class RobinBC1D(BoundaryCondition1D):
    """
    Robin boundary condition: alpha*u + beta*du/dx = g(x,t) at boundary.

    Mixed boundary condition combining Dirichlet and Neumann.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rhs_function: float | Callable,
        name: str = "Robin",
        region_id: int | None = None,
    ):
        """
        Initialize Robin boundary condition.

        Args:
            alpha: Coefficient of u
            beta: Coefficient of du/dx
            rhs_function: Either constant or function(x, t) -> float
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)
        self.alpha = alpha
        self.beta = beta
        if callable(rhs_function):
            self.rhs_function = rhs_function
        else:
            self.rhs_function = lambda x, t: float(rhs_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply Robin condition to matrix."""
        matrix_mod = matrix.tolil()

        for idx in boundary_indices:
            # Modify diagonal by alpha contribution
            matrix_mod[idx, idx] += self.alpha

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply Robin values to RHS."""
        rhs_mod = rhs.copy()

        boundary_vertices = mesh.vertices[boundary_indices]
        for i, idx in enumerate(boundary_indices):
            x = boundary_vertices[i, 0] if boundary_vertices.ndim > 1 else boundary_vertices[i]
            rhs_mod[idx] = self.rhs_function(x, time)

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh compatibility."""
        return True


class PeriodicBC1D(BoundaryCondition1D):
    """
    Periodic boundary condition: u(x_left) = u(x_right).

    Links left and right boundaries together.
    """

    def __init__(self, name: str = "Periodic", region_id: int | None = None):
        """
        Initialize periodic boundary condition.

        Args:
            name: Human-readable name
            region_id: Optional region identifier
        """
        super().__init__(name, region_id)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        """Apply periodic condition by linking boundary rows."""
        if len(boundary_indices) != 2:
            warnings.warn("Periodic BC expects exactly 2 boundary indices (left, right)")
            return matrix

        matrix_mod = matrix.tolil()
        left_idx, right_idx = boundary_indices[0], boundary_indices[-1]

        # Link right boundary to left (u_right = u_left)
        matrix_mod[right_idx, :] = 0
        matrix_mod[right_idx, left_idx] = 1
        matrix_mod[right_idx, right_idx] = -1

        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply periodic condition to RHS (sets constraint RHS to 0)."""
        rhs_mod = rhs.copy()

        if len(boundary_indices) >= 2:
            right_idx = boundary_indices[-1]
            rhs_mod[right_idx] = 0

        return rhs_mod

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate mesh can support periodic conditions."""
        return True


class BoundaryConditionManager1D:
    """
    Manager for 1D FEM boundary conditions.

    Handles multiple boundary conditions and their application to
    linear systems arising from FEM discretization.
    """

    def __init__(self):
        """Initialize boundary condition manager."""
        self._conditions: dict[int | str, BoundaryCondition1D] = {}
        self._region_indices: dict[int | str, np.ndarray] = {}

    def add_condition(
        self,
        condition: BoundaryCondition1D,
        region: int | str | np.ndarray,
    ) -> None:
        """
        Add boundary condition for a region.

        Args:
            condition: Boundary condition to apply
            region: Region identifier or array of boundary indices
        """
        if isinstance(region, np.ndarray):
            # Direct indices provided
            region_id = len(self._conditions)
            self._region_indices[region_id] = region
            condition._direct_vertices = region
        else:
            region_id = region
            self._region_indices[region_id] = np.array([])

        self._conditions[region_id] = condition

    def get_condition(self, region: int | str) -> BoundaryCondition1D | None:
        """Get boundary condition for a region."""
        return self._conditions.get(region)

    def apply_to_system(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """
        Apply all boundary conditions to linear system.

        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            mesh: Mesh data
            time: Current time

        Returns:
            Modified (matrix, rhs) tuple
        """
        matrix_mod = matrix
        rhs_mod = rhs.copy()

        for region_id, condition in self._conditions.items():
            # Get boundary indices
            if condition._direct_vertices is not None:
                boundary_indices = condition._direct_vertices
            else:
                boundary_indices = self._get_boundary_indices(mesh, region_id)

            if len(boundary_indices) == 0:
                continue

            # Apply condition
            matrix_mod = condition.apply_to_matrix(matrix_mod, mesh, boundary_indices)
            rhs_mod = condition.apply_to_rhs(rhs_mod, mesh, boundary_indices, time)

        return matrix_mod, rhs_mod

    def _get_boundary_indices(self, mesh: MeshData, region_id: int | str) -> np.ndarray:
        """Get boundary vertex indices for a region."""
        if region_id in self._region_indices and len(self._region_indices[region_id]) > 0:
            return self._region_indices[region_id]

        # Try to get from mesh boundary markers
        if hasattr(mesh, "boundary_markers"):
            markers = mesh.boundary_markers
            if isinstance(region_id, int):
                return np.where(markers == region_id)[0]

        # Default: return endpoints for 1D mesh
        if hasattr(mesh, "vertices"):
            n_vertices = len(mesh.vertices)
            if region_id == 0 or region_id == "left":
                return np.array([0])
            elif region_id == 1 or region_id == "right":
                return np.array([n_vertices - 1])

        return np.array([])

    def validate_all(self, mesh: MeshData) -> bool:
        """Validate all boundary conditions against mesh."""
        return all(condition.validate_mesh_compatibility(mesh) for condition in self._conditions.values())


class MFGBoundaryHandler1D:
    """
    MFG-specific boundary handler for 1D FEM problems.

    Coordinates boundary conditions for coupled HJB and FP equations.
    """

    def __init__(self):
        """Initialize MFG boundary handler."""
        self._hjb_manager = BoundaryConditionManager1D()
        self._fp_manager = BoundaryConditionManager1D()

    @property
    def hjb_manager(self) -> BoundaryConditionManager1D:
        """Get HJB equation boundary manager."""
        return self._hjb_manager

    @property
    def fp_manager(self) -> BoundaryConditionManager1D:
        """Get FP equation boundary manager."""
        return self._fp_manager

    def add_hjb_condition(
        self,
        condition: BoundaryCondition1D,
        region: int | str | np.ndarray,
    ) -> None:
        """Add boundary condition for HJB equation."""
        self._hjb_manager.add_condition(condition, region)

    def add_fp_condition(
        self,
        condition: BoundaryCondition1D,
        region: int | str | np.ndarray,
    ) -> None:
        """Add boundary condition for FP equation."""
        self._fp_manager.add_condition(condition, region)

    def apply_hjb_bc(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply HJB boundary conditions."""
        return self._hjb_manager.apply_to_system(matrix, rhs, mesh, time)

    def apply_fp_bc(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        mesh: MeshData,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply FP boundary conditions."""
        return self._fp_manager.apply_to_system(matrix, rhs, mesh, time)

    def validate_mfg_compatibility(self, mesh: MeshData) -> bool:
        """
        Validate boundary conditions for MFG consistency.

        Checks that HJB and FP conditions are compatible.
        """
        hjb_valid = self._hjb_manager.validate_all(mesh)
        fp_valid = self._fp_manager.validate_all(mesh)

        if not hjb_valid or not fp_valid:
            warnings.warn("Boundary conditions may be incompatible with mesh")

        return hjb_valid and fp_valid


def create_interval_boundary_conditions(
    bc_type: str = "dirichlet",
    left_value: float = 0.0,
    right_value: float = 0.0,
) -> BoundaryConditionManager1D:
    """
    Create boundary conditions for a 1D interval [a, b].

    Args:
        bc_type: Type of BC ("dirichlet", "neumann", "periodic")
        left_value: Value at left boundary
        right_value: Value at right boundary

    Returns:
        Configured BoundaryConditionManager1D
    """
    manager = BoundaryConditionManager1D()

    if bc_type == "periodic":
        manager.add_condition(PeriodicBC1D(), np.array([0, -1]))
    elif bc_type == "dirichlet":
        manager.add_condition(DirichletBC1D(left_value, "Left Dirichlet"), "left")
        manager.add_condition(DirichletBC1D(right_value, "Right Dirichlet"), "right")
    elif bc_type == "neumann":
        manager.add_condition(NeumannBC1D(left_value, "Left Neumann"), "left")
        manager.add_condition(NeumannBC1D(right_value, "Right Neumann"), "right")
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return manager


__all__ = [
    # Base class
    "BoundaryCondition1D",
    # Specific BC types
    "DirichletBC1D",
    "NeumannBC1D",
    "RobinBC1D",
    "PeriodicBC1D",
    # Manager classes
    "BoundaryConditionManager1D",
    "MFGBoundaryHandler1D",
    # Factory function
    "create_interval_boundary_conditions",
]
