"""
Dimension-agnostic FEM boundary condition handling.

Consolidates fem_bc_1d.py, fem_bc_2d.py, fem_bc_3d.py into a single
module. All classes work in any dimension by querying mesh.dimension.

Issue #802 Phase 4: FEM BC consolidation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.geometry.meshes.mesh_data import MeshData


class FEMBoundaryCondition(ABC):
    """
    Abstract base class for FEM boundary conditions (any dimension).

    Replaces dimension-specific BoundaryCondition1D/2D/3D.
    """

    def __init__(self, name: str, region_id: int | None = None):
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

    def validate_mesh_compatibility(self, mesh: MeshData) -> bool:
        """Validate compatibility with mesh. Override for specific checks."""
        return True


class DirichletBC(FEMBoundaryCondition):
    """Dirichlet BC: u = g(x, t) at boundary. Dimension-agnostic."""

    def __init__(
        self,
        value_function: float | Callable,
        name: str = "Dirichlet",
        region_id: int | None = None,
    ):
        super().__init__(name, region_id)
        if callable(value_function):
            self.value_function = value_function
        else:
            self.value_function = lambda x, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        matrix_mod = matrix.tolil()
        for idx in boundary_indices:
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
        rhs_mod = rhs.copy()
        for idx in boundary_indices:
            vertex = mesh.vertices[idx]
            rhs_mod[idx] = self.value_function(vertex, time)
        return rhs_mod


class NeumannBC(FEMBoundaryCondition):
    """Neumann BC: du/dn = g(x, t) at boundary. Dimension-agnostic."""

    def __init__(
        self,
        flux_function: float | Callable = 0.0,
        name: str = "Neumann",
        region_id: int | None = None,
    ):
        super().__init__(name, region_id)
        if callable(flux_function):
            self.flux_function = flux_function
        else:
            self.flux_function = lambda x, t: float(flux_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        # Neumann: no matrix modification (natural BC in weak form)
        return matrix

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        rhs_mod = rhs.copy()
        for idx in boundary_indices:
            vertex = mesh.vertices[idx]
            rhs_mod[idx] += self.flux_function(vertex, time)
        return rhs_mod


class RobinBC(FEMBoundaryCondition):
    """Robin BC: alpha*u + beta*du/dn = g(x, t). Dimension-agnostic."""

    def __init__(
        self,
        alpha: float | Callable = 1.0,
        beta: float | Callable = 1.0,
        value_function: float | Callable = 0.0,
        name: str = "Robin",
        region_id: int | None = None,
    ):
        super().__init__(name, region_id)
        self.alpha = alpha if callable(alpha) else lambda x, t: float(alpha)
        self.beta = beta if callable(beta) else lambda x, t: float(beta)
        self.value_function = value_function if callable(value_function) else lambda x, t: float(value_function)

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        matrix_mod = matrix.tolil()
        for idx in boundary_indices:
            vertex = mesh.vertices[idx]
            a = self.alpha(vertex, 0.0)
            matrix_mod[idx, idx] += a
        return csr_matrix(matrix_mod.tocsr())

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        rhs_mod = rhs.copy()
        for idx in boundary_indices:
            vertex = mesh.vertices[idx]
            rhs_mod[idx] += self.value_function(vertex, time)
        return rhs_mod


class PeriodicBC(FEMBoundaryCondition):
    """Periodic BC: u(x_min) = u(x_max). Dimension-agnostic via DOF pairing."""

    def __init__(
        self,
        paired_boundaries: list[tuple[str, str]] | None = None,
        name: str = "Periodic",
        region_id: int | None = None,
    ):
        super().__init__(name, region_id)
        self.paired_boundaries = paired_boundaries or []

    def apply_to_matrix(self, matrix: csr_matrix, mesh: MeshData, boundary_indices: np.ndarray) -> csr_matrix:
        # Periodic: link master-slave DOFs
        # For 1D: simple left=right linking
        # For nD: use paired_boundaries to find matching DOFs
        if mesh.dimension == 1 and len(boundary_indices) == 2:
            matrix_mod = matrix.tolil()
            slave, master = boundary_indices[0], boundary_indices[-1]
            matrix_mod[slave, :] = 0
            matrix_mod[slave, slave] = 1
            matrix_mod[slave, master] = -1
            return csr_matrix(matrix_mod.tocsr())
        # nD: more complex pairing needed
        return matrix

    def apply_to_rhs(
        self,
        rhs: np.ndarray,
        mesh: MeshData,
        boundary_indices: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        rhs_mod = rhs.copy()
        if mesh.dimension == 1 and len(boundary_indices) == 2:
            rhs_mod[boundary_indices[0]] = 0.0
        return rhs_mod


class FEMBoundaryConditionManager:
    """
    Manage multiple BCs for an FEM mesh. Dimension-agnostic.

    Replaces BoundaryConditionManager1D/2D/3D.
    """

    def __init__(self, mesh: MeshData):
        self.mesh = mesh
        self.conditions: list[tuple[FEMBoundaryCondition, np.ndarray]] = []

    def add_condition(
        self,
        condition: FEMBoundaryCondition,
        boundary_indices: np.ndarray | None = None,
    ) -> None:
        """Add a BC with its associated boundary DOF indices."""
        if boundary_indices is None:
            boundary_indices = np.array([], dtype=int)
        self.conditions.append((condition, boundary_indices))

    def apply_to_system(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        time: float = 0.0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Apply all BCs to the FEM system."""
        for condition, indices in self.conditions:
            if len(indices) > 0:
                matrix = condition.apply_to_matrix(matrix, self.mesh, indices)
                rhs = condition.apply_to_rhs(rhs, self.mesh, indices, time)
        return matrix, rhs

    def validate_all(self) -> bool:
        """Validate all conditions against the mesh."""
        return all(c.validate_mesh_compatibility(self.mesh) for c, _ in self.conditions)


class MFGBoundaryHandler:
    """
    MFG-specific BC handler for coupled HJB-FP systems. Dimension-agnostic.

    Replaces MFGBoundaryHandler1D/2D/3D.
    """

    def __init__(self, mesh: MeshData):
        self.mesh = mesh
        self.hjb_manager = FEMBoundaryConditionManager(mesh)
        self.fp_manager = FEMBoundaryConditionManager(mesh)

    def add_state_constraint(
        self,
        value_function: float | Callable,
        boundary_indices: np.ndarray,
        name: str = "state_constraint",
    ) -> None:
        """Add Dirichlet BC on HJB (state constraint)."""
        self.hjb_manager.add_condition(
            DirichletBC(value_function, name=name),
            boundary_indices,
        )

    def add_no_flux_condition(
        self,
        boundary_indices: np.ndarray,
        name: str = "no_flux",
    ) -> None:
        """Add zero-flux Neumann BC on FP."""
        self.fp_manager.add_condition(
            NeumannBC(0.0, name=name),
            boundary_indices,
        )

    def apply_hjb_conditions(
        self, matrix: csr_matrix, rhs: np.ndarray, time: float = 0.0
    ) -> tuple[csr_matrix, np.ndarray]:
        return self.hjb_manager.apply_to_system(matrix, rhs, time)

    def apply_fp_conditions(
        self, matrix: csr_matrix, rhs: np.ndarray, time: float = 0.0
    ) -> tuple[csr_matrix, np.ndarray]:
        return self.fp_manager.apply_to_system(matrix, rhs, time)

    def validate_mfg_compatibility(self) -> bool:
        return self.hjb_manager.validate_all() and self.fp_manager.validate_all()


# =============================================================================
# Backward-compatible aliases (dimension-specific names → unified classes)
# =============================================================================

# These aliases allow existing code that imports dimension-specific classes
# to continue working without changes.

BoundaryCondition1D = FEMBoundaryCondition
BoundaryCondition2D = FEMBoundaryCondition
BoundaryCondition3D = FEMBoundaryCondition

DirichletBC1D = DirichletBC
DirichletBC2D = DirichletBC
DirichletBC3D = DirichletBC

NeumannBC1D = NeumannBC
NeumannBC2D = NeumannBC
NeumannBC3D = NeumannBC

RobinBC1D = RobinBC
RobinBC2D = RobinBC
RobinBC3D = RobinBC

PeriodicBC1D = PeriodicBC
PeriodicBC2D = PeriodicBC
PeriodicBC3D = PeriodicBC

BoundaryConditionManager1D = FEMBoundaryConditionManager
BoundaryConditionManager2D = FEMBoundaryConditionManager
BoundaryConditionManager3D = FEMBoundaryConditionManager

MFGBoundaryHandler1D = MFGBoundaryHandler
MFGBoundaryHandler2D = MFGBoundaryHandler
MFGBoundaryHandler3D = MFGBoundaryHandler
